from abc import ABC, abstractmethod

import json
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import f1_score

from tqdm.auto import tqdm
from sklearn.model_selection import PredefinedSplit, GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt

RANDOM_STATE = 42

from utils.embedding_transformer_utils import df_type_from_name, get_torch_device, slug
from utils.modeling_helpers import (
    compute_binary_pos_weight,
    evaluate_classifier_predictions,
    make_tensor_loader,
    prepare_scaled_tabular_features,
)
from utils.textual_utils.config import PROJECT_ROOT

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_JOBS = -1 if torch.cuda.is_available() else 1

def _is_cuda_device(device) -> bool:
    return str(device).lower().startswith("cuda")

try:
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models in the pipeline.
    Ensures that every model implements mandatory preprocessing and prediction logic.
    """
    def __init__(self, model_name: str, model=None):
        self.model_name = model_name
        self.model = model

    @abstractmethod
    def preprocess(self, data: pd.DataFrame, is_training=True) -> tuple:
        """
        Cleaning, feature engineering, and/or scaling required for the model.
        :param data: The raw input dataframe.
        :param is_training: Boolean to distinguish between training and inference (e.g., for fit_transform vs transform).
        :return: Tuple (X, y) ready for the model.
        """
        pass

    def train(self, X_train, y_train, **kwargs):
        """
        Fits the model to the training data.
        """
        if self.model is None:
            raise ValueError(f"Model for {self.model_name} is not initialized.")
        print(f"[{self.model_name}] Starting training...")
        self.model.fit(X_train, y_train, **kwargs)

    def grid_search(self):
        raise Exception('NO GRID SEARCH DEFINED!!')

    def hypertune_pipeline(self, df_train, df_val, param_grid, n_jobs=N_JOBS, frac=0.01, **kwargs):
        """ 
        Hypertune, find the best parameters.
        :param df_train: Dataframe with train features and targets.
        :param df_val: Dtaaframe with validation features and targets.
        :param param_grid: params to tune.
        """
        # Guard against accidental forwarding of training-only args to GridSearchCV.
        frac = kwargs.pop("frac", frac)
        grid_search = self.grid_search(df_train, df_val, param_grid, n_jobs=n_jobs, **kwargs)
        # final training creating model with best params
        best_params = grid_search.best_params_
        self.model.set_params(**best_params)
        print(f'[{self.model_name}] Train model with best params...')
        self.train_pipeline(df_train, frac=frac)

    def predict(self, X):
        """
        Inference: Uses the trained model to make predictions.
        """
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Calculates and prints performance metrics.
        """
        display_labels = getattr(self.model, "classes_", None)
        evaluate_classifier_predictions(y_true, y_pred, display_labels=display_labels)


    def train_pipeline(self, raw_train, frac=0.01, random_state=RANDOM_STATE, **kwargs):
        """
        Complete pipeline for train:
        - preprocess for training
        - fitting the model
        - prediction
        - evaluation
        """
        X_tr, y_tr = self.preprocess(raw_train, is_training=True, **kwargs)
        self.train(X_tr, y_tr)

        if frac < 1:
            new_size = int(len(X_tr) * frac)
            print(f"Selected {new_size}/{len(X_tr)}")
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X_tr), size=new_size, replace=False)

            X_sample = X_tr[idx]
            y_sample = y_tr.iloc[idx] if hasattr(y_tr, "iloc") else y_tr[idx]

            X_tr = X_sample
            y_tr = y_sample
        elif frac > 1:
            raise ValueError("Frac value not Valid (should be < 1)")
    

        y_pred = self.predict(X_tr)
        self.evaluate(y_tr, y_pred)

    def test_pipeline(self, raw_test, **kwargs):
        """
        Complete pipeline for test:
        - preprocess for training
        - prediction
        - evaluation
        """
        X_te, y_te = self.preprocess(raw_test, is_training=False, **kwargs)

        y_pred = self.predict(X_te)
        self.evaluate(y_te, y_pred)


class KNNModel(BaseModel):
    """
    Implementation of the KNN Baseline model using paper features.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="KNN", n_jobs=N_JOBS, **kwargs):
        # Initialize the scikit-learn KNN model
        knn_internal = KNeighborsClassifier(
            n_jobs=n_jobs,
            **kwargs
        )
        super().__init__(model_name=model_name, model=knn_internal)
        
        # We need to keep track of the scaler to apply the same transformation in test
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def grid_search(self, df_train, df_val, param_grid, max_tuning_samples=50000, n_jobs=N_JOBS, **kwargs) -> list:
        """ 
        Hypertune, find the best parameters.
        :param X_val: Dataframe with validation features.
        :param y_val: Dtaaframe with validation targets.
        :param param_grid: params to tune.
        """
        print(f'[{self.model_name}] Grid Search...')
        # Downsampling for speed
        def downsample_indices(indices, n_samples):
            if len(indices) > n_samples:
                return np.random.choice(indices, n_samples, replace=False)
            return indices
        
        X_train_scaled, y_train = self.preprocess(df_train, is_training=True)
        X_val_scaled, y_val = self.preprocess(df_val, is_training=False)

        # sample for tuning
        train_tuning_idx = downsample_indices(np.arange(len(X_train_scaled)), int(max_tuning_samples * 0.8))
        val_tuning_idx = downsample_indices(np.arange(len(X_val_scaled)), int(max_tuning_samples * 0.2))

        X_subset = np.vstack((X_train_scaled[train_tuning_idx], X_val_scaled[val_tuning_idx]))
        y_subset = np.concatenate((y_train.iloc[train_tuning_idx], y_val.iloc[val_tuning_idx]))

        # Create split indices: -1 for train, 0 for validation
        split_index = np.concatenate([-1 * np.ones(len(train_tuning_idx)), 0 * np.ones(len(val_tuning_idx))])
        ps = PredefinedSplit(test_fold=split_index)

        # GridSearchCV
        print(f"\nStarting tuning on {len(X_subset)} samples...")
        grid_search = GridSearchCV(
            KNeighborsClassifier(n_jobs=n_jobs), 
            param_grid=param_grid, 
            cv=ps,
            n_jobs=n_jobs, 
            **kwargs
        )
        grid_search.fit(X_subset, y_subset)

        # print results
        best_params = grid_search.best_params_
        print("\nBest parameters found:")
        print(best_params)

        # Final training on the full dataset with the best parameters
        model = grid_search.best_estimator_
        model.fit(X_train_scaled, y_train)

        print(f"\nOptimal model ready: {model}")

        return grid_search

    def predict_proba(self, X):
        """
        Get probability scores (useful for AUC calculation).
        """
        return self.model.predict_proba(X)

class XGBModel(BaseModel):
    """ 
    Implementation of the XGB Baseline model using paper features.
    Inherits from BaseModel.
    """
    def __init__(self, model_name='XGB', device=DEVICE, **kwargs):
        # Initialize the scikit-learn KNN model
        xgb_internal = XGBClassifier(
            device=device,
            **kwargs
        )
        super().__init__(model_name=model_name, model=xgb_internal)
        
        # We need to keep track of the scaler to apply the same transformation in test
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            verbose=verbose,
            model_name=self.model_name,
        )

    def grid_search(self, df_train, df_val, param_grid, device=DEVICE, n_jobs=N_JOBS, **kwargs):
        """
        Hypertune, find the best parameters.
        :param X_val: Dataframe with validation features.
        :param y_val: Dtaaframe with validation targets.
        :param param_grid: params to tune.
        """

        print(f'[{self.model_name}] Grid Search...')
        # preprocess data
        X_train_scaled, y_train = self.preprocess(df_train, is_training=True)
        X_val_scaled, y_val = self.preprocess(df_val, is_training=False)

        search_n_jobs = 1 if _is_cuda_device(device) else n_jobs
        if _is_cuda_device(device) and n_jobs != 1:
            print(
                f"[{self.model_name}] CUDA detected: using n_jobs=1 for RandomizedSearchCV "
                "to avoid running multiple GPU fits at the same time."
            )

        model_params = self.model.get_params()
        model_params.update({
            "tree_method": model_params.get("tree_method") or "hist",
            "device": device,
        })
        if _is_cuda_device(device):
            model_params["n_jobs"] = 1

        model = XGBClassifier(**model_params)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_jobs=search_n_jobs,
            **kwargs
        )

        # Run randomized search on the validation split (kept small by earlier sampling)
        random_search.fit(X_val_scaled, y_val)

        best_params = random_search.best_params_
        print("\nBest parameters found:")
        print(best_params)

        print(f"\nOptimal model ready: {random_search.best_estimator_}")

        return random_search


    def predict_proba(self, X):
        """
        Get probability scores (useful for AUC calculation).
        """
        return self.model.predict_proba(X)

class LGBModel(BaseModel):
    """ 
    Implementation of the LightGBM model for large-scale embedding classification.
    Optimized for high speed and parallelization.
    """
    def __init__(self, model_name='LGBM', device=DEVICE, n_jobs=N_JOBS, random_state=RANDOM_STATE, **kwargs):
        # Initialize LightGBM Classifier
        # device can be 'cpu' or 'gpu'
        lgb_internal = lgb.LGBMClassifier(
            device=device,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        super().__init__(model_name=model_name, model=lgb_internal)
        self.scaler = RobustScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        return prepare_scaled_tabular_features(
            data,
            self.scaler,
            is_training=is_training,
            as_dataframe=True,
            verbose=verbose,
            model_name=self.model_name,
        )

    def grid_search(self, df_train, df_val, param_grid, n_iter=15, n_jobs=N_JOBS, **kwargs):
        """
        Hyperparameter tuning using RandomizedSearchCV for efficiency.
        """
        print(f'[{self.model_name}] Starting Randomized Search...')
        X_train_scaled, y_train = self.preprocess(df_train, is_training=True)
        X_val_scaled, y_val = self.preprocess(df_val, is_training=False)

        # We use a subset for tuning to speed up the process, 
        # but LGBM is fast enough to handle larger chunks than KNN
        model = lgb.LGBMClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            **kwargs
        )

        # Tuning on validation to find best params quickly
        random_search.fit(X_val_scaled, y_val)

        print(f"\nBest parameters: {random_search.best_params_}")
        
        # Final training on the full training set
        self.model = random_search.best_estimator_
        self.model.fit(X_train_scaled, y_train)
        
        return random_search

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class PairEmbeddingTransformer(nn.Module if nn is not None else object):
    """
    Lightweight Transformer encoder for article/reference embedding pairs.

    Input shape: [batch, 2, embedding_dim]
    - token 0: article embedding
    - token 1: reference embedding
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for PairEmbeddingTransformer: {_TORCH_IMPORT_ERROR}")
        super().__init__()
        self.input_projection = nn.Linear(embedding_dim, d_model)
        self.type_embedding = nn.Parameter(torch.zeros(1, 2, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, X):
        X = self.input_projection(X) + self.type_embedding
        cls = self.cls_token.expand(X.size(0), -1, -1)
        X = torch.cat([cls, X], dim=1)
        X = self.encoder(X)
        cls_output = self.norm(X[:, 0])
        return self.classifier(cls_output).squeeze(-1)


class PairEmbeddingTransformerModel(BaseModel):
    """
    Transformer model wrapper that follows the same pipeline shape as KNNModel.

    The class owns:
    - preprocessing and scaling
    - training loop
    - prediction and probability output
    - evaluation
    - checkpoint saving
    """
    def __init__(self, model_name: str = "pair_embedding_transformer_128", device=None, **model_params):
        if torch is None or nn is None:
            raise OSError(f"PyTorch is required for PairEmbeddingTransformerModel: {_TORCH_IMPORT_ERROR}")
        if "embedding_dim" not in model_params:
            raise ValueError("embedding_dim must be specified in model_params for PairEmbeddingTransformerModel")
        self.model_params = {
            "embedding_dim": model_params["embedding_dim"],
            "d_model": model_params.get("d_model", model_params["embedding_dim"]),
            "nhead": model_params.get("nhead", 8),
            "num_layers": model_params.get("num_layers", 2),
            "dim_feedforward": model_params.get("dim_feedforward", model_params["embedding_dim"] * 2),
            "dropout": model_params.get("dropout", 0.15),
        }
        self.model_params.update(model_params)
        self.device = torch.device(device) if device is not None else get_torch_device()
        transformer = PairEmbeddingTransformer(**self.model_params).to(self.device)
        super().__init__(model_name=model_name, model=transformer)

        self.scaler = RobustScaler()
        self.article_cols: list[str] | None = None
        self.ref_cols: list[str] | None = None
        self.history: list[dict] = []
        self.threshold = 0.5
        self.last_metrics: dict = {}

    def _ensure_embedding_columns(self, data: pd.DataFrame) -> None:
        if self.article_cols is None:
            self.article_cols = sorted([col for col in data.columns if col.startswith("article_emb_")])
        if self.ref_cols is None:
            self.ref_cols = sorted([col for col in data.columns if col.startswith("ref_emb_")])
        if len(self.article_cols) != self.model_params["embedding_dim"] or len(self.ref_cols) != self.model_params["embedding_dim"]:
            raise ValueError(
                f"Expected {self.model_params['embedding_dim']} article/ref embedding columns, found "
                f"{len(self.article_cols)} and {len(self.ref_cols)}"
            )

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose: bool = True) -> tuple:
        """
        Prepare embedding pairs as [n_rows, 2, embedding_dim].

        The scaler is fit only on training data and reused for validation/test,
        matching the behavior of KNNModel.
        """
        print(f"[{self.model_name}] Preprocessing data...")
        self._ensure_embedding_columns(data)

        article = data[self.article_cols].to_numpy(dtype=np.float32)
        reference = data[self.ref_cols].to_numpy(dtype=np.float32)
        flat_features = np.concatenate([article, reference], axis=1)
        y = data["is_reference_valid"].to_numpy(dtype=np.float32)

        if is_training:
            flat_features = self.scaler.fit_transform(flat_features)
        else:
            flat_features = self.scaler.transform(flat_features)

        embedding_dim = self.model_params["embedding_dim"]
        X = flat_features.astype(np.float32).reshape(-1, 2, embedding_dim)

        if verbose:
            print("Label distribution:")
            print(pd.Series(y).value_counts(normalize=True))

        return X, y

    def _make_loader(self, X, y=None, batch_size: int = 512, shuffle: bool = False):
        return make_tensor_loader(
            torch,
            TensorDataset,
            DataLoader,
            X,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def _compute_pos_weight(self, y):
        return compute_binary_pos_weight(torch, y, self.device)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 5,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 2,
        **kwargs,
    ):
        """
        Train the PyTorch transformer.

        Validation is optional. When provided, the best checkpoint is selected
        by weighted F1 score.
        """
        print(f"[{self.model_name}] Starting training...")
        train_loader = self._make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._compute_pos_weight(y_train))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_f1 = -np.inf
        best_state = None
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            row = {"epoch": epoch, "train_loss": float(total_loss / len(train_loader.dataset))}

            if X_val is not None and y_val is not None:
                val_prob = self.predict_proba(X_val, batch_size=batch_size)
                val_pred = (val_prob >= self.threshold).astype(int)
                val_f1 = f1_score(np.asarray(y_val).astype(int), val_pred, average="weighted")
                row["val_f1_weighted"] = float(val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.history.append(row)
            print(row)

            if X_val is not None and y_val is not None and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X, batch_size: int = 512):
        """
        Return positive-class probabilities.
        """
        self.model.eval()
        loader = self._make_loader(X, batch_size=batch_size, shuffle=False)
        probabilities = []
        with torch.no_grad():
            for (X_batch,) in tqdm(loader, desc="predict", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                logits = self.model(X_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, X, batch_size: int = 512):
        return (self.predict_proba(X, batch_size=batch_size) >= self.threshold).astype(int)

    def evaluate(self, y_true, y_pred, title: str | None = None):
        """
        Print classification metrics and plot a confusion matrix.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        metrics = evaluate_classifier_predictions(y_true, y_pred, display_labels=[0, 1], output_dict=True)
        if title:
            plt.title(title)
            plt.show()

        self.last_metrics = metrics
        return metrics

    def train_pipeline(self, raw_train, raw_val=None, **kwargs):
        """
        Complete train pipeline:
        - preprocess train
        - preprocess validation with train scaler
        - fit transformer
        - evaluate on train
        """
        verbose = kwargs.pop("verbose", True)
        X_tr, y_tr = self.preprocess(raw_train, is_training=True, verbose=verbose)
        if raw_val is not None:
            X_val, y_val = self.preprocess(raw_val, is_training=False, verbose=False)
        else:
            X_val, y_val = None, None

        self.train(X_tr, y_tr, X_val=X_val, y_val=y_val, **kwargs)
        y_pred = self.predict(X_tr, batch_size=kwargs.get("batch_size", 512))
        return self.evaluate(y_tr, y_pred, title=f"{self.model_name} - Train confusion matrix")

    def test_pipeline(self, raw_test, batch_size: int = 512, **kwargs):
        """
        Complete test pipeline:
        - preprocess test with train scaler
        - predict
        - evaluate
        """
        X_te, y_te = self.preprocess(raw_test, is_training=False, **kwargs)
        y_pred = self.predict(X_te, batch_size=batch_size)
        return self.evaluate(y_te, y_pred, title=f"{self.model_name} - Test confusion matrix")


class GraphFeatureTransformer(nn.Module if nn is not None else object):
    """
    Lightweight Transformer encoder for tabular graph features.

    Input shape: [batch, num_features]
    Each scalar feature is projected as a token, enriched with a learned feature
    embedding, and summarized through a CLS token.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
    ):
        if nn is None:
            raise OSError(f"PyTorch is required for GraphFeatureTransformer: {_TORCH_IMPORT_ERROR}")
        if num_features <= 0:
            raise ValueError("num_features must be greater than 0")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, num_features, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, max(d_model // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 1), 1),
        )

        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, X):
        X = X.unsqueeze(-1)
        X = self.value_projection(X) + self.feature_embedding
        cls = self.cls_token.expand(X.size(0), -1, -1)
        X = torch.cat([cls, X], dim=1)
        X = self.encoder(X)
        cls_output = self.norm(X[:, 0])
        return self.classifier(cls_output).squeeze(-1)


class SimpleTransformer(BaseModel):
    """
    Transformer model wrapper for numeric tabular features.

    It follows the same public API as the other model classes:
    - preprocess
    - train / train_pipeline
    - predict / predict_proba
    - evaluate / test_pipeline
    """
    def __init__(
        self,
        model_name: str = "SimpleTransformer_graph",
        device=None,
        feature_cols: list[str] | None = None,
        **model_params,
    ):
        if torch is None or nn is None:
            raise OSError(f"PyTorch is required for SimpleTransformer: {_TORCH_IMPORT_ERROR}")

        self.model_params = {
            "num_features": model_params.get("num_features"),
            "d_model": model_params.get("d_model", 64),
            "nhead": model_params.get("nhead", 4),
            "num_layers": model_params.get("num_layers", 2),
            "dim_feedforward": model_params.get("dim_feedforward", 128),
            "dropout": model_params.get("dropout", 0.15),
        }
        self.device = torch.device(device) if device is not None else get_torch_device()
        self.scaler = RobustScaler()
        self.feature_cols = feature_cols
        self.history: list[dict] = []
        self.threshold = 0.5
        self.last_metrics: dict = {}

        transformer = None
        if self.model_params["num_features"] is not None:
            transformer = GraphFeatureTransformer(**self.model_params).to(self.device)

        super().__init__(model_name=model_name, model=transformer)

    def _init_model(self, num_features: int) -> None:
        if self.model is not None:
            return
        self.model_params["num_features"] = num_features
        self.model = GraphFeatureTransformer(**self.model_params).to(self.device)

    def _infer_feature_columns(self, data: pd.DataFrame) -> list[str]:
        drop_cols = [
            "is_reference_valid",
            "article_id",
            "ref_id",
            "vector_text_article",
            "vector_text_ref",
            "split",
        ]
        candidate = data.drop(columns=drop_cols, errors="ignore").copy()
        numeric_cols = candidate.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric feature columns found for SimpleTransformer")
        return numeric_cols

    def _ensure_feature_columns(self, data: pd.DataFrame, is_training: bool) -> None:
        if self.feature_cols is None:
            if not is_training:
                raise ValueError("SimpleTransformer must be fitted before preprocessing validation/test data")
            self.feature_cols = self._infer_feature_columns(data)

        missing_cols = [col for col in self.feature_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        expected_features = self.model_params.get("num_features")
        if expected_features is not None and len(self.feature_cols) != expected_features:
            raise ValueError(
                f"Expected {expected_features} feature columns, found {len(self.feature_cols)}"
            )
        self._init_model(len(self.feature_cols))

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose: bool = True) -> tuple:
        """
        Prepare numeric features as a float matrix [n_rows, n_features].
        """
        if verbose:
            print(f"[{self.model_name}] Preprocessing {len(data)} rows...")

        self._ensure_feature_columns(data, is_training=is_training)

        X = data[self.feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = data["is_reference_valid"].to_numpy(dtype=np.float32)

        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        if verbose:
            print("Label distribution:")
            print(pd.Series(y).value_counts(normalize=True))

        return X_scaled.astype(np.float32), y

    def _make_loader(self, X, y=None, batch_size: int = 512, shuffle: bool = False):
        return make_tensor_loader(
            torch,
            TensorDataset,
            DataLoader,
            X,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def _compute_pos_weight(self, y):
        return compute_binary_pos_weight(torch, y, self.device)

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 5,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 2,
        **kwargs,
    ):
        """
        Train the tabular-feature transformer.
        """
        if self.model is None:
            self._init_model(X_train.shape[1])

        print(f"[{self.model_name}] Starting training...")
        train_loader = self._make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._compute_pos_weight(y_train))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_f1 = -np.inf
        best_state = None
        epochs_without_improvement = 0
        self.history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for X_batch, y_batch in tqdm(train_loader, desc=f"epoch {epoch} train", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * X_batch.size(0)

            row = {"epoch": epoch, "train_loss": float(total_loss / len(train_loader.dataset))}

            if X_val is not None and y_val is not None:
                val_prob = self.predict_proba(X_val, batch_size=batch_size)
                val_pred = (val_prob >= self.threshold).astype(int)
                val_f1 = f1_score(np.asarray(y_val).astype(int), val_pred, average="weighted")
                row["val_f1_weighted"] = float(val_f1)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            self.history.append(row)
            print(row)

            if X_val is not None and y_val is not None and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_proba(self, X, batch_size: int = 512):
        """
        Return positive-class probabilities.
        """
        if self.model is None:
            raise ValueError("SimpleTransformer model is not initialized.")

        self.model.eval()
        loader = self._make_loader(X, batch_size=batch_size, shuffle=False)
        probabilities = []
        with torch.no_grad():
            for (X_batch,) in tqdm(loader, desc="predict", leave=False):
                X_batch = X_batch.to(self.device, non_blocking=True)
                logits = self.model(X_batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, X, batch_size: int = 512):
        return (self.predict_proba(X, batch_size=batch_size) >= self.threshold).astype(int)

    def evaluate(self, y_true, y_pred, title: str | None = None):
        """
        Print classification metrics and plot a confusion matrix.
        """
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)

        metrics = evaluate_classifier_predictions(y_true, y_pred, display_labels=[0, 1], output_dict=True)
        if title:
            plt.title(title)
            plt.show()

        self.last_metrics = metrics
        return metrics

    def train_pipeline(self, raw_train, raw_val=None, **kwargs):
        """
        Complete train pipeline:
        - preprocess train graph features
        - preprocess optional validation graph features with train scaler
        - fit transformer
        - evaluate on train
        """
        verbose = kwargs.pop("verbose", True)
        X_tr, y_tr = self.preprocess(raw_train, is_training=True, verbose=verbose)
        if raw_val is not None:
            X_val, y_val = self.preprocess(raw_val, is_training=False, verbose=False)
        else:
            X_val, y_val = None, None

        self.train(X_tr, y_tr, X_val=X_val, y_val=y_val, **kwargs)
        y_pred = self.predict(X_tr, batch_size=kwargs.get("batch_size", 512))
        return self.evaluate(y_tr, y_pred, title=f"{self.model_name} - Train confusion matrix")

    def test_pipeline(self, raw_test, batch_size: int = 512, **kwargs):
        """
        Complete test pipeline:
        - preprocess test graph features with train scaler
        - predict
        - evaluate
        """
        X_te, y_te = self.preprocess(raw_test, is_training=False, **kwargs)
        y_pred = self.predict(X_te, batch_size=batch_size)
        return self.evaluate(y_te, y_pred, title=f"{self.model_name} - Test confusion matrix")

    def save_model(
        self,
        params=None,
        df_name="graph_features",
        model_family="transformer",
        split_name="predefined_train_validation_test",
        summary=None,
        force=False,
    ):
        """
        Save a PyTorch checkpoint plus a JSON summary.
        """
        if self.model is None:
            raise ValueError("SimpleTransformer model is not initialized.")

        params_to_save = params if params is not None else self.model_params
        summary = summary if summary is not None else self.last_metrics

        df_type = df_type_from_name(df_name)
        model_slug = slug(self.model_name)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        artifact_dir = PROJECT_ROOT / "Models" / df_type / model_slug
        artifact_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifact_dir / f"{model_slug}__{timestamp}.pt"
        summary_path = artifact_dir / f"{model_slug}__{timestamp}.json"

        if model_path.exists() and not force:
            raise FileExistsError(f"Model already exists: {model_path}")

        checkpoint = {
            "model_state_dict": self.model.cpu().state_dict(),
            "model_params": self.model_params,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "threshold": self.threshold,
            "history": self.history,
        }
        torch.save(checkpoint, model_path)
        self.model.to(self.device)

        payload = {
            "timestamp": timestamp,
            "df_type": df_type,
            "df_name": df_name,
            "model_family": model_family,
            "model_name": self.model_name,
            "split_name": split_name,
            "params": params_to_save,
            "model_path": str(model_path),
            "performance": summary,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        print("Hyperparameters:", params_to_save)
        print(f"Saved {self.model_name} model to:", model_path)
        print("Saved summary to:", summary_path)
        return model_path, summary_path

SimpleTransformerModel = SimpleTransformer

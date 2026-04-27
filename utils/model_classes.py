from abc import ABC, abstractmethod

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils.model_saver import save_model_artifact
from sklearn.preprocessing import StandardScaler

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

    def predict(self, X):
        """
        Inference: Uses the trained model to make predictions.
        """
        return self.model.predict(X)
    
    def evaluate(self, y_true, y_pred):
        """
        Calculates and prints performance metrics.
        """
        # print(f"\n--- Evaluation for {self.model_name} ---")
        # print(classification_report(y_true, y_pred))
        # print("Confusion Matrix:")
        # print(confusion_matrix(y_true, y_pred))

        print(classification_report(y_true, y_pred, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot(cmap="Blues")


    def train_pipeline(self, raw_train, **kwargs):
        """
        Complete pipeline for train:
        - preprocess for training
        - fitting the model
        - prediction
        - evaluation
        """
        X_tr, y_tr = self.preprocess(raw_train, is_training=True, **kwargs)
        self.train(X_tr, y_tr)
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

    def save_model(self, params=None,
                    df_name='df', model_family='', split_name='predefined_train_validation_test', summary={}, force=False):
        """ 
        Save the trained model in a file, to reuse it.
        """
        params_to_save = params if params is not None else self.model.get_params()
        model_path, summary_path = save_model_artifact(
            model=self.model,
            df_name=df_name,
            model_family=model_family,
            model_name=self.model_name,
            split_name=split_name,
            params=params_to_save,
            summary=summary,
            force=force,
        )
        print("Hyperparameters:", params_to_save)
        print(f"Saved {self.model_name} model to:", model_path)
        print("Saved summary to:", summary_path)



class KNNModel(BaseModel):
    """
    Implementation of the KNN Baseline model using paper embeddings.
    Inherits from BaseModel.
    """
    def __init__(self, model_name="KNN", n_jobs=-1, **kwargs):
        # Initialize the scikit-learn KNN model
        knn_internal = KNeighborsClassifier(
            n_jobs=n_jobs,
            **kwargs
        )
        super().__init__(model_name=model_name, model=knn_internal)
        
        # We need to keep track of the scaler to apply the same transformation in test
        self.scaler = StandardScaler()

    def preprocess(self, data: pd.DataFrame, is_training: bool = True, verbose=True) -> tuple:
        """
        Prepares features by concatenating article and reference embeddings.
        :param data: Dataframe containing 'embedding_article' and 'embedding_ref' columns.
        :param is_training: If True, fits the scaler. If False, only transforms.
        """
        print(f"[{self.model_name}] Preprocessing data...")

        # 1. drop columns that are not features (handling names based on your notebook)
        drop_cols = ["is_reference_valid", "article_id", "ref_id", "vector_text_article", "vector_text_ref", "split"]
        X = data.drop(columns=drop_cols, errors="ignore").copy()
        y = data["is_reference_valid"].copy()

        # 2. Scaling
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        if verbose:
            print("Label distribution:")
            print(y.value_counts(normalize=True))

        return X_scaled, y        

    def predict_proba(self, X):
        """
        Get probability scores (useful for AUC calculation).
        """
        return self.model.predict_proba(X)

    
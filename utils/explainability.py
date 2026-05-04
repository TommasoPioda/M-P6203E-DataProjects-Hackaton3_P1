from lime.lime_tabular import LimeTabularExplainer
import shap
from IPython.display import display, HTML
import torch
from pathlib import Path 
import joblib 
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

def lime_explainer(X_train, X_test, y_test, model):
    """
    Based on train and test data, it use the LIME to
    explain locally a valid and invalid reference.
    
    Args:
        X_train (pd.DataFrame): Dataframe of the train set
        
        X_test (pd.DataFrame): Dataframe of the test set
        
        y_test (pd.Series): Series containing target values of the test set
        
        model: Model trained over the data
        
        
    Plot a valid and invalid reference, explaining locally
    which features push the model to take such decisions.       
    """

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        mode="classification"
        )

    valid_ref = X_test[y_test==1].iloc[0].values
    invalid_ref = X_test[y_test==0].iloc[0].values

    #explain valid reference
    print("Valid reference:")
    exp_val = explainer.explain_instance(
        valid_ref,
        model.predict_proba
        )

    display(HTML(exp_val.as_html()))

    #explain valid reference
    print("\nInvalid reference")
    exp_inval = explainer.explain_instance(
        invalid_ref,
        model.predict_proba
        )

    display(HTML(exp_inval.as_html()))

def shap_tree_explainer(X_test, model):
    """
    Based on test data, it use the SHAP to
    explain a tree model globally, showing
    which features influence positively and negatively
    the decision.
    
    Args:
        X_test (pd.DataFrame): Dataframe of the test set
        
        model (TreeModel): Tree model trained over the data
        
        
    Plot a summary and a bar plot over the test set    
    """

    
    explainer = shap.TreeExplainer(model)
    shap_val = explainer.shap_values(X_test)

    # summary plot
    shap.summary_plot(shap_val, X_test)

    # bar plot
    shap.summary_plot(shap_val, X_test, plot_type="bar")

def shap_kernel_explainer(X_train, X_test, model):
    """
    Based on test data, it use the SHAP to
    explain a transformer model globally, showing
    which features influence positively and negatively
    the decision.
    
    Args:
        X_test (pd.DataFrame): Dataframe of the train set
        
        X_test (pd.DataFrame): Dataframe of the test set
        
        model (Transformer): Transformer model trained over the data
        
    Plot a summary and a bar plot over the test set    
    """    

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def predict_fn(data_numpy):
        with torch.no_grad():
            inputs = torch.tensor(data_numpy).float().to(device)
            outputs = model(inputs)
            return torch.sigmoid(outputs).cpu().numpy()

    if device == "cuda":
        train_data = X_train.iloc[:200]
        test_data = X_test.iloc[:100]
    else:
        train_data = X_train.iloc[:100]
        test_data = X_test.iloc[:50]
        
    explainer = shap.KernelExplainer(predict_fn, train_data.values)

    shap_val = explainer.shap_values(test_data.values)

    # summary plot
    shap.summary_plot(shap_val, test_data.values)
    
    # bar plot
    shap.summary_plot(shap_val, test_data, plot_type="bar")
    
def load_all_models(models_dir: Path, selected_model: str = None) -> dict[str, object]:
    """Loads all .pkl/.joblib models present in a folder."""
    if not models_dir.exists():
        print(f"Models folder not found: {models_dir}")
        return {}

    model_files = sorted(
        p for p in models_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pkl", ".joblib"}
    )

    models = {}
    for model_path in model_files:
        try:
            models[model_path.stem] = joblib.load(model_path)
        except Exception as exc:
            print(f"Unable to load {model_path.name}: {exc}")
            

    if selected_model:
        model = {name: model for name, model in models.items() if selected_model in name}
        print(f"Selected model: {model}")
        return model

    print(f"Models loaded from {models_dir.name}: {len(models)}")
    for name in models:
        print(f" - {name}")
    return models

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns:
        split = df["split"].astype(str).str.lower()
        train_df = df[split.isin(["train", "training"])].copy()
        test_df = df[split.isin(["test", "validation", "val"])].copy()
        if len(train_df) and len(test_df):
            return train_df, test_df

    cut = int(len(df) * 0.8)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def load_textual_split(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    train_df, test_df = split_train_test(df)
    print(f"{path.name}: train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


def load_graph_split(graph_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(graph_dir / "train.parquet")
    test_df = pd.read_parquet(graph_dir / "test.parquet")
    print(f"graph features: train={train_df.shape}, test={test_df.shape}")
    return train_df, test_df


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = [
        "is_reference_valid",
        "article_id",
        "ref_id",
        "vector_text_article",
        "vector_text_ref",
        "split",
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    return X.select_dtypes(include=["number", "bool"]).columns.tolist()


class ProbabilityAdapter:
    """Adapts single-column/3D array output to the format expected by LIME."""
    def __init__(self, model, reshape_to: tuple[int, ...] | None = None):
        self.model = model
        self.reshape_to = reshape_to

    def predict_proba(self, X):
        X_array = np.asarray(X)
        if self.reshape_to is not None and X_array.ndim == 2:
            X_array = X_array.reshape((len(X_array), *self.reshape_to))

        proba = np.asarray(self.model.predict_proba(X_array))
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        elif proba.ndim == 2 and proba.shape[1] == 1:
            positive = proba[:, 0]
            proba = np.column_stack([1 - positive, positive])
        return proba


def prepared_explainability_data(model, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Prepares X/y consistent with the preprocessing saved inside the model."""
    if hasattr(model, "preprocess"):
        X_train, y_train = model.preprocess(train_df, is_training=False, verbose=False)
        X_test, y_test = model.preprocess(test_df, is_training=False, verbose=False)
    else:
        cols = raw_feature_columns(train_df)
        X_train = train_df[cols].copy()
        X_test = test_df[cols].copy()
        y_train = train_df["is_reference_valid"].copy()
        y_test = test_df["is_reference_valid"].copy()

    reshape_to = None
    if np.asarray(X_train).ndim == 3:
        shape = np.asarray(X_train).shape
        reshape_to = shape[1:]
        X_train = np.asarray(X_train).reshape(shape[0], -1)
        X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], -1)

    if hasattr(model, "article_cols") and getattr(model, "article_cols", None):
        feature_names = list(model.article_cols) + list(model.ref_cols)
    elif isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = raw_feature_columns(train_df)
        if len(feature_names) != np.asarray(X_train).shape[1]:
            feature_names = [f"feature_{i:03d}" for i in range(np.asarray(X_train).shape[1])]

    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=train_df.index)
    X_test_df = pd.DataFrame(X_test, columns=feature_names, index=test_df.index)
    y_train = pd.Series(y_train, index=train_df.index, name="is_reference_valid")
    y_test = pd.Series(y_test, index=test_df.index, name="is_reference_valid")

    return X_train_df, X_test_df, y_train, y_test, ProbabilityAdapter(model, reshape_to=reshape_to)


def sample_for_explainability(X: pd.DataFrame, y: pd.Series, max_rows: int = 5_000, random_state:int = 42):
    if max_rows is None or len(X) <= max_rows:
        return X, y

    y = pd.Series(y, index=X.index)
    required_idx = y.groupby(y).head(1).index
    remaining_idx = y.index.difference(required_idx)
    n_remaining = max(max_rows - len(required_idx), 0)

    if n_remaining > 0:
        sampled_idx = remaining_idx.to_series().sample(
            n=min(n_remaining, len(remaining_idx)),
            random_state=random_state,
        ).index
        idx = required_idx.union(sampled_idx)
    else:
        idx = required_idx[:max_rows]

    return X.loc[idx], y.loc[idx]


def tree_model_for_shap(model):
    candidate = getattr(model, "model", model)
    class_name = candidate.__class__.__name__.lower()
    tree_keywords = ["xgb", "lgb", "randomforest", "decisiontree", "gradientboost", "catboost"]
    if any(keyword in class_name for keyword in tree_keywords):
        return candidate
    return None

def transformer_model_for_shap(model):
    candidate = getattr(model, "model", model)
    class_name = candidate.__class__.__name__.lower()
    transformer_keywords = ["transformer", "simple_transformer"]
    if any(keyword in class_name for keyword in transformer_keywords):
        return candidate
    return None


def run_explainability_block(title: str, models: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame, expl_sample_size:int=5_000, random_state:int=42):
    display(Markdown(f"## {title}"))

    if not models:
        print("No models available for this block.")
        return

    for model_name, model in models.items():
        print("" + "=" * 90)
        print(f"Explainability for model: {model_name}")

        try:
            X_train, X_test, y_train, y_test, lime_model = prepared_explainability_data(model, train_df, test_df)
            X_train_lime, y_train_lime = sample_for_explainability(X_train, y_train, max_rows=expl_sample_size, random_state=random_state)
            X_test_lime, y_test_lime = sample_for_explainability(X_test, y_test, max_rows=expl_sample_size, random_state=random_state)
            print(f"Features used: {X_train_lime.shape[1]}")
            print(f"LIME sample train/test: {X_train_lime.shape}, {X_test_lime.shape}")
        except Exception as exc:
            print(f"Data preparation failed: {exc}")
            continue

        try:
            lime_explainer(X_train_lime, X_test_lime, y_test_lime, lime_model)
            print("LIME completed")
        except Exception as exc:
            print(f"LIME failed: {exc}")

        shap_model = tree_model_for_shap(model)

        try:
            shap_tree_explainer(X_test_lime, shap_model)
            print("SHAP tree completed")
        except Exception as exc:
            print(f"SHAP tree skipped/failed: {exc}")
            

        shap_model = transformer_model_for_shap(model)
        if shap_model is None:
            print("SHAP tree skipped: the model is not a transformer model supported by KernelExplainer.")
            continue

        try:
            shap_kernel_explainer(X_train_lime, X_test_lime, shap_model)
            print("SHAP transformer completed")
        except Exception as exc:
            print(f"SHAP transformer skipped/failed: {exc}")
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


DEFAULT_NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "is_reference_valid",
    "article_id",
    "ref_id",
    "vector_text_article",
    "vector_text_ref",
    "split",
)


def split_features_target(
    data: pd.DataFrame,
    target_col: str = "is_reference_valid",
    drop_cols: Sequence[str] = DEFAULT_NON_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target while preserving the existing column contract."""
    if target_col not in data.columns:
        raise KeyError(f"Column '{target_col}' not found in DataFrame.")

    X = data.drop(columns=list(drop_cols), errors="ignore").copy()
    y = data[target_col].copy()
    return X, y


def prepare_scaled_tabular_features(
    data: pd.DataFrame,
    scaler,
    *,
    is_training: bool,
    as_dataframe: bool = False,
    verbose: bool = True,
    model_name: str = "model",
) -> tuple[np.ndarray | pd.DataFrame, pd.Series]:
    """Apply the common tabular preprocessing used by KNN, XGB and LGBM wrappers."""
    if verbose:
        print(f"[{model_name}] Preprocessing {len(data)} rows...")

    X, y = split_features_target(data)
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X) if is_training else scaler.transform(X)

    if verbose:
        print("Label distribution:")
        print(y.value_counts(normalize=True))

    if as_dataframe:
        return pd.DataFrame(X_scaled, columns=feature_names, index=data.index), y
    return X_scaled, y


def evaluate_classifier_predictions(
    y_true,
    y_pred,
    *,
    display_labels=None,
    cmap: str = "Blues",
    output_dict: bool = False,
) -> dict | None:
    """Print a classification report and plot the corresponding confusion matrix."""
    print(classification_report(y_true, y_pred, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(cmap=cmap)

    if not output_dict:
        return None

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {
        "accuracy": float(report["accuracy"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
    }


def make_tensor_loader(torch_module, tensor_dataset_cls, data_loader_cls, X, y=None, *, batch_size: int, shuffle: bool, pin_memory: bool):
    """Create a PyTorch DataLoader from numpy arrays without duplicating wrapper code."""
    X_tensor = torch_module.tensor(X, dtype=torch_module.float32)
    if y is None:
        dataset = tensor_dataset_cls(X_tensor)
    else:
        y_tensor = torch_module.tensor(y, dtype=torch_module.float32)
        dataset = tensor_dataset_cls(X_tensor, y_tensor)

    return data_loader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )


def compute_binary_pos_weight(torch_module, y, device):
    """Return BCEWithLogitsLoss positive-class weight for imbalanced binary labels."""
    y_array = np.asarray(y, dtype=np.float32)
    num_pos = float(y_array.sum())
    num_neg = float(len(y_array) - num_pos)
    return torch_module.tensor(num_neg / max(num_pos, 1.0), dtype=torch_module.float32, device=device)

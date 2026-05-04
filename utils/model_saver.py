from __future__ import annotations

import datetime as _datetime
import pickle
import re
from pathlib import Path
from typing import Any


__all__ = ["save_model_artifact"]


def save_model_artifact(
    model: Any,
    df_name: str | None = None,
    model_name: str | None = None,
    relative_model_dir: str | Path | None = None,
) -> Path:
    """Save a pickle model using the legacy notebook-facing layout.

    This function preserves the original top-level ``utils.model_saver`` API:
    ``Models/<df_name>/<model_name>_<timestamp>.pkl``.
    """
    if df_name is None:
        raise ValueError("df_name is required.")
    if model_name is None:
        raise ValueError("model_name is required.")
    if relative_model_dir is None:
        raise ValueError("relative_model_dir is required.")

    cleaned_df_name = re.sub(r"\.parquet$", "", str(df_name))
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = Path(relative_model_dir) / cleaned_df_name / f"{model_name}_{timestamp}.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_save_path, "wb") as handle:
        pickle.dump(model, handle)

    return model_save_path

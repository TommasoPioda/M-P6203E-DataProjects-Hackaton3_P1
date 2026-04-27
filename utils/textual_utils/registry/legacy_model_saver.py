from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from ..config import PROJECT_ROOT


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "default"


def _df_type_from_name(df_name: str) -> str:
    value = _slug(df_name)
    if any(token in value for token in ("embedding", "textual")):
        return "embeddings"
    if "graph" in value:
        return "graph_based"
    if any(token in value for token in ("mix", "mixed", "hybrid")):
        return "mix"
    if any(token in value for token in ("normal", "classic", "raw", "exploded")):
        return "normal"
    return value or "normal"


def save_model_artifact(
    model: Any,
    *,
    df_name: str,
    model_family: str,
    model_name: str,
    split_name: str,
    params: dict[str, Any] | None = None,
    cv_results: Any | None = None,
    tokenizer: Any | None = None,
    summary: dict[str, Any] | None = None,
    force: bool = False,
    root: Path | None = None,
) -> tuple[Path, Path]:
    """Save model artifacts under Models/<df_type>/<model_name>.

    Returns a tuple: (model_artifact_path, metadata_json_path).
    """
    params = params or {}
    summary = summary or {}

    df_type = _df_type_from_name(df_name)
    model_slug = _slug(model_name)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    saved_name = f"{model_slug}__{timestamp}"

    base_dir = root or (PROJECT_ROOT / "Models")
    artifact_dir = base_dir / df_type / model_slug
    artifact_dir.mkdir(parents=True, exist_ok=True)

    is_transformer = hasattr(model, "save_pretrained")

    if is_transformer:
        model_path = artifact_dir / saved_name
        if model_path.exists() and not force:
            raise FileExistsError(f"Model already exists: {model_path}")
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(model_path)
    else:
        model_path = artifact_dir / f"{saved_name}.joblib"
        if model_path.exists() and not force:
            raise FileExistsError(f"Model already exists: {model_path}")
        joblib.dump(model, model_path)

    cv_results_path: Path | None = None
    if cv_results is not None:
        try:
            import pandas as pd  # local import to avoid hard dependency during module import

            cv_df = pd.DataFrame(cv_results)
            cv_results_path = artifact_dir / f"{saved_name}__cv_results.csv"
            cv_df.to_csv(cv_results_path, index=False)
        except Exception:
            cv_results_path = None

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "df_type": df_type,
        "df_name": df_name,
        "model_family": model_family,
        "model_name": model_name,
        "split_name": split_name,
        "params": params,
        "model_path": str(model_path),
        "performance": summary,
    }
    if cv_results_path is not None:
        payload["cv_results_path"] = str(cv_results_path)

    metadata_path = artifact_dir / f"{saved_name}.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    return model_path, metadata_path


def save_sklearn_model(
    estimator: Any,
    *,
    feature_set: str,
    experiment_name: str,
    model_name: str | None = None,
    cv_results_path: Path | None = None,
    summary: dict | None = None,
    keep_latest: int = 5,
    force: bool = False,
    runtime_copy: bool = False,
    root: Path | None = None,
) -> Path:
    """Save a scikit-learn style estimator with the shared artifact layout."""
    estimator_name = model_name or estimator.__class__.__name__
    model_path, _ = save_model_artifact(
        estimator,
        df_name=feature_set,
        model_family=experiment_name,
        model_name=estimator_name,
        split_name="all",
        params=getattr(estimator, "get_params", lambda: {})(),
        summary=summary,
        force=force,
        root=root,
    )

    # Backward compatibility: optional runtime alias and external cv results copy.
    if runtime_copy:
        try:
            runtime_path = model_path.parent / f"{_slug(estimator_name)}__runtime.joblib"
            joblib.dump(estimator, runtime_path)
        except Exception:
            pass

    if cv_results_path is not None:
        try:
            if Path(cv_results_path).exists():
                legacy_copy = model_path.parent / Path(cv_results_path).name
                if legacy_copy.resolve() != Path(cv_results_path).resolve():
                    legacy_copy.write_bytes(Path(cv_results_path).read_bytes())
        except Exception:
            pass

    return model_path


def save_classic_model(
    estimator: Any,
    cv_results_path: Path | None = None,
    summary: dict | None = None,
    model_name: str | None = None,
    keep_latest: int = 5,
    force: bool = False,
    runtime_copy: bool = False,
):
    """Backward-compatible wrapper for existing notebook calls."""
    return save_sklearn_model(
        estimator,
        feature_set="classic",
        experiment_name="classic_ml",
        model_name=model_name,
        cv_results_path=cv_results_path,
        summary=summary,
        keep_latest=keep_latest,
        force=force,
        runtime_copy=runtime_copy,
    )


def save_transformer_model(
    model: Any,
    *,
    feature_set: str,
    experiment_name: str,
    tokenizer: Any | None = None,
    summary: dict | None = None,
    force: bool = False,
    root: Path | None = None,
) -> Path:
    """Save a Hugging Face style transformer model with the shared artifact layout."""
    if not hasattr(model, "save_pretrained"):
        raise TypeError("save_transformer_model expects a Hugging Face style model with save_pretrained().")

    model_path, _ = save_model_artifact(
        model,
        df_name=feature_set,
        model_family=experiment_name,
        model_name=model.__class__.__name__,
        split_name="all",
        params={},
        tokenizer=tokenizer,
        summary=summary,
        force=force,
        root=root,
    )
    return model_path

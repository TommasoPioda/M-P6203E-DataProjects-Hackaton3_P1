from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _slug(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "default"


def _params_hash(obj: Any) -> str:
    try:
        if hasattr(obj, "get_params"):
            params = obj.get_params()
        elif hasattr(obj, "config") and hasattr(obj.config, "to_dict"):
            params = obj.config.to_dict()
        else:
            params = obj.__dict__
        encoded = json.dumps(params, sort_keys=True, default=str)
    except Exception:
        encoded = repr(obj)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _artifact_dir(feature_set: str, experiment_name: str, root: Path | None = None) -> Path:
    base_dir = root or (PROJECT_ROOT / "Models" / "embedding_based")
    return base_dir / _slug(feature_set) / _slug(experiment_name)


def _compact_params(params: dict[str, Any] | None, max_items: int = 4, max_key_len: int = 10, max_val_len: int = 12) -> str:
    if not params:
        return "default"
    parts: list[str] = []
    for key in sorted(params.keys())[:max_items]:
        value = params[key]
        if isinstance(value, float):
            rendered = f"{value:.4g}"
        else:
            rendered = str(value)
        key_slug = _slug(str(key))[:max_key_len]
        val_slug = _slug(rendered)[:max_val_len]
        parts.append(f"{key_slug}-{val_slug}")
    compact = "__".join(parts) or "default"
    return compact[:80]


def _model_split_dir(
    *,
    df_name: str,
    model_family: str,
    model_name: str,
    split_name: str | None,
    root: Path | None = None,
) -> Path:
    base_dir = root or (PROJECT_ROOT / "Models" / "embedding_based")
    # Keep folder names informative but short enough for Windows path limits.
    split_name = split_name or "all"
    return base_dir / df_name / model_family / model_name / split_name


def save_model_artifact(
    model: Any,
    *,
    df_name: str,
    model_family: str,
    model_name: str,
    split_name: str | None = None,
    params: dict[str, Any] | None = None,
    cv_results: Any | None = None,
    tokenizer: Any | None = None,
    summary: dict[str, Any] | None = None,
    force: bool = False,
    root: Path | None = None,
) -> tuple[Path, Path]:
    """Save model artifacts under models/<df>/<family>/<model>/<split>.

    Returns a tuple: (model_artifact_path, metadata_json_path).
    """
    artifact_dir = _model_split_dir(
        df_name=df_name,
        model_family=model_family,
        model_name=model_name,
        split_name=split_name,
        root=root,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    params = params or {}
    params_compact = _compact_params(params)
    params_hash = _params_hash(params)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    model_slug = _slug(model_name)[:16]

    is_transformer = hasattr(model, "save_pretrained")
    if is_transformer:
        model_path = artifact_dir / f"{model_slug}__{params_compact}__{params_hash}"
        if model_path.exists() and force:
            for child in model_path.glob("**/*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(model_path)
    else:
        model_path = artifact_dir / f"{model_slug}__{params_compact}__{params_hash}.joblib"
        # Additional safety net for Windows path length: fallback to a shorter filename.
        if len(str(model_path)) > 240:
            model_path = artifact_dir / f"{model_slug}__{params_hash}.joblib"
        if (not model_path.exists()) or force:
            joblib.dump(model, model_path)

    cv_results_path: Path | None = None
    if cv_results is not None:
        try:
            import pandas as pd  # local import to avoid hard dependency during module import

            cv_df = pd.DataFrame(cv_results)
            cv_results_path = artifact_dir / f"cv_results__{model_slug}__{timestamp}.csv"
            cv_df.to_csv(cv_results_path, index=False)
        except Exception:
            cv_results_path = None

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "df_name": df_name,
        "model_family": model_family,
        "model_name": model_name,
        "split_name": split_name,
        "params": params,
        "params_hash": params_hash,
        "model_path": str(model_path),
    }
    if cv_results_path is not None:
        payload["cv_results_path"] = str(cv_results_path)
    if summary:
        payload.update(summary)

    metadata_path = artifact_dir / f"summary__{model_slug}__{timestamp}.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

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
    """Save a scikit-learn style estimator under a feature-aware folder.

    The saved path becomes models/<feature_set>/<experiment_name>/... so textual and
    non-textual feature runs remain separated.
    """
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
    """Save a Hugging Face style transformer model in a feature-aware folder.

    The model is saved into a stable "latest" directory for the experiment so repeated
    runs overwrite the same artifact instead of accumulating duplicates.
    """
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

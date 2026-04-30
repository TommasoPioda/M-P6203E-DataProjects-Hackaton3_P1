from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


def slug(value: str) -> str:
    """Return a filesystem-safe lowercase name."""
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "default"


def df_type_from_name(df_name: str) -> str:
    """Map a dataframe name to the artifact folder used under Models/."""
    value = slug(df_name)
    if any(token in value for token in ("embedding", "textual")):
        return "embeddings"
    if "graph" in value:
        return "graph_based"
    if any(token in value for token in ("mix", "mixed", "hybrid")):
        return "mix"
    if any(token in value for token in ("normal", "classic", "raw", "exploded")):
        return "normal"
    return value or "normal"


def get_torch_device():
    """Return CUDA when available, otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_torch_seed(seed: int = 42) -> None:
    """Set seeds for random, numpy and torch."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_dataframe(data: pd.DataFrame, max_samples: int | None, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic sample, or the full dataframe when max_samples is None."""
    if max_samples is None or len(data) <= max_samples:
        return data.reset_index(drop=True)
    return data.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking up from the current notebook directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "setup.py").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Project root not found. Run from inside the repository.")


def load_pair_embedding_transformer_model(checkpoint_path: str | Path, model_name: str = "pair_embedding_transformer_128", device=None):
    """Load a PairEmbeddingTransformerModel checkpoint saved by the model wrapper."""
    import torch
    from utils.model_classes import PairEmbeddingTransformerModel

    checkpoint_path = Path(checkpoint_path)
    device = device or get_torch_device()

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    loaded_model = PairEmbeddingTransformerModel(
        model_name=model_name,
        device=device,
        **checkpoint["model_params"],
    )
    loaded_model.model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.scaler = checkpoint["scaler"]
    loaded_model.article_cols = checkpoint["article_cols"]
    loaded_model.ref_cols = checkpoint["ref_cols"]
    loaded_model.threshold = checkpoint.get("threshold", 0.5)
    loaded_model.history = checkpoint.get("history", [])
    loaded_model.model.eval()
    return loaded_model


def load_simple_transformer_model(checkpoint_path: str | Path, model_name: str = "SimpleTransformer_graph", device=None):
    """Load a SimpleTransformer checkpoint saved by the model wrapper."""
    import torch
    from utils.model_classes import SimpleTransformer

    checkpoint_path = Path(checkpoint_path)
    device = device or get_torch_device()

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    loaded_model = SimpleTransformer(
        model_name=model_name,
        device=device,
        feature_cols=checkpoint["feature_cols"],
        **checkpoint["model_params"],
    )
    loaded_model.model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.scaler = checkpoint["scaler"]
    loaded_model.threshold = checkpoint.get("threshold", 0.5)
    loaded_model.history = checkpoint.get("history", [])
    loaded_model.model.eval()
    return loaded_model

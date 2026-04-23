from pathlib import Path

# __file__ is utils/textual_utils/config.py, parent is utils/textual_utils/, parent.parent.parent is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define main directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "Models"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
UTILS_DIR = PROJECT_ROOT / "utils"

# Sub-paths
RAW_DATA_FILE = DATA_DIR / "DBLP-Citation-network-V18" / "DBLP-Citation-network-V18.jsonl"
PARQUET_DIR = DATA_DIR / "parquet"
EMBEDDINGS_DIR = DATA_DIR / "textual_features" / "embeddings_128d"

def ensure_dirs():
    for directory in [DATA_DIR, PARQUET_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_saved_classic_dir() -> Path:
    """Return the existing saved_classic directory, normalizing between 'Models' and 'models'.

    Ensures the directory exists and returns a Path object that other code should use
    when writing model artifacts.
    """
    candidates = [MODELS_DIR / "saved_classic", PROJECT_ROOT / "models" / "saved_classic"]
    # Do NOT create directories here. Return the first existing candidate, or
    # a canonical path if none exists. Caller may create it explicitly if desired.
    for p in candidates:
        if p.exists():
            return p
    # Fallback: return canonical path without creating it
    return MODELS_DIR / "saved_classic"

ensure_dirs()

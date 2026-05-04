import json
import uuid
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
except ImportError:
    PreTrainedModel, PreTrainedTokenizer = None, None

class LocalModelRegistry:
    """Local model registry that stores experiments in an organized layout."""
    def __init__(self, registry_base_path: str = None):
        if registry_base_path is None:
            # Store artifacts at the project root, next to the notebooks folder.
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            self.base_path = PROJECT_ROOT / "Models" / "embedding_based"
        else:
            self.base_path = Path(registry_base_path)
            
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_id}"

    def save_experiment(
        self, 
        model_name: str, 
        model: Any, 
        metrics: Dict[str, float] = None, 
        params: Dict[str, Any] = None, 
        dataset_info: Dict[str, str] = None, 
        tokenizer: Union['PreTrainedTokenizer', None] = None
    ) -> Path:
        metrics = metrics or {}
        params = params or {}
        dataset_info = dataset_info or {}
        
        run_id = self._generate_run_id()
        run_dir = self.base_path / model_name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "run_id": run_id,
            "model_name": model_name,
            "saved_at": datetime.now().isoformat(),
            "metrics": metrics,
            "parameters": params,
            "dataset_info": dataset_info
        }
        
        with open(run_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        if PreTrainedModel and isinstance(model, PreTrainedModel):
            model.save_pretrained(run_dir)
            if tokenizer:
                tokenizer.save_pretrained(run_dir)
        elif hasattr(model, "save_model"): 
            model.save_model(run_dir / "xgboost_model.json")
        else:
            joblib.dump(model, run_dir / "model.joblib")
            
        print(f"Model artifact saved successfully in: {run_dir}")
        return run_dir

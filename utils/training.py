from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, random_split
from transformers import Trainer, TrainerCallback, TrainingArguments

from .citation_dataset import BertCitationDataset
from .feature_extractor import build_classic_ml_matrix


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_ratio: float = 0.8, seed: int = 42):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def compute_pos_weight(labels) -> torch.Tensor:
    labels_array = np.asarray(labels, dtype=np.float32)
    num_pos = float(labels_array.sum())
    num_neg = float(len(labels_array) - num_pos)
    weight = num_neg / max(num_pos, 1.0)
    return torch.tensor(weight, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class PlotLossCallback(TrainerCallback):
    def __init__(self, save_dir: str = "./plots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.loss_history_file = os.path.join(save_dir, "loss_history.pkl")
        plt.ioff()
        if os.path.exists(self.loss_history_file):
            with open(self.loss_history_file, "rb") as handle:
                self.loss_history = pickle.load(handle)
        else:
            self.loss_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        self.loss_history.append(logs["loss"])
        plt.clf()
        plt.plot(self.loss_history, label="Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "training_loss.png"))

        with open(self.loss_history_file, "wb") as handle:
            pickle.dump(self.loss_history, handle)


def create_training_arguments(
    output_dir: str,
    per_device_train_batch_size: int = 96,
    per_device_eval_batch_size: int = 64,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    logging_steps: int = 20,
    save_total_limit: int = 2,
    report_to: list[str] | None = None,
    fp16: bool | None = None,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        report_to=report_to or [],
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available() if fp16 is None else fp16,
    )


def build_bert_datasets(
    df_training,
    tokenizer,
    max_len: int = 128,
    train_ratio: float = 0.8,
    seed: int = 42,
    include_authors: bool = True,
) -> tuple[Subset, Subset, torch.Tensor]:
    dataset = BertCitationDataset(df_training, tokenizer, max_len=max_len, include_authors=include_authors)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=train_ratio, seed=seed)
    pos_weight = compute_pos_weight(df_training["is_reference_valid"].to_numpy())
    return train_dataset, val_dataset, pos_weight


def predict_with_grade(text1: str, text2: str, model, tokenizer, device, max_len: int = 128) -> dict:
    model.eval()
    inputs = tokenizer(
        text1,
        text2,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()

    probability = torch.sigmoid(logits).item()
    predicted_class = int(probability > 0.5)
    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "grade": get_grade_from_probability(probability),
    }


def get_grade_from_probability(probability: float) -> str:
    if probability > 0.85:
        return "Excellent"
    if probability > 0.65:
        return "Good"
    if probability > 0.40:
        return "Weak"
    return "Poor"


def evaluate_predictions(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
    }


def predict_trainer_outputs(trainer: Trainer, eval_dataset, offset: int = 0, max_eval_samples: int | None = None, threshold: float = 0.5) -> dict:
    if isinstance(eval_dataset, Subset):
        subset = eval_dataset
    elif max_eval_samples is not None:
        subset = Subset(eval_dataset, list(range(offset, min(offset + max_eval_samples, len(eval_dataset)))))
    else:
        subset = eval_dataset

    pred_output = trainer.predict(subset)
    logits = pred_output.predictions.reshape(-1)
    y_prob = 1 / (1 + np.exp(-logits))
    metrics = evaluate_predictions(pred_output.label_ids, y_prob, threshold=threshold)
    metrics["y_prob"] = y_prob
    metrics["label_ids"] = pred_output.label_ids
    return metrics


def rolling_accuracy(y_true, y_pred, block_size: int = 100, num_samples: int = 50, seed: int = 42) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    rng = np.random.default_rng(seed)
    scores = []
    if len(y_true) < block_size:
        return np.array(scores, dtype=np.float32)

    for _ in range(num_samples):
        start_idx = rng.integers(0, len(y_true) - block_size + 1)
        block_true = y_true[start_idx : start_idx + block_size]
        block_pred = y_pred[start_idx : start_idx + block_size]
        scores.append(accuracy_score(block_true, block_pred))
    return np.asarray(scores, dtype=np.float32)


def split_train_test(
    X,
    y,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
):
    stratify_y = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )


def incremental_sgd_step(
    model,
    df_training,
    first_run: bool,
    random_state: int,
    test_size: float = 0.1,
    max_features: int = 256,
    include_authors: bool = True,
    classes: tuple[int, int] = (0, 1),
) -> dict:
    X_model, y_model, feature_extractor, artifacts = build_classic_ml_matrix(
        df_training,
        max_features=max_features,
        include_authors=include_authors,
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X_model,
        y_model,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    if first_run:
        model.partial_fit(X_train, y_train, classes=list(classes))
        first_run = False
    else:
        model.partial_fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float((y_test == y_pred).mean())

    return {
        "model": model,
        "first_run": first_run,
        "accuracy": accuracy,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_model": X_model,
        "y_model": y_model,
        "feature_extractor": feature_extractor,
        "artifacts": artifacts,
    }


def build_classification_report(
    y_true,
    y_pred,
    target_names: list[str] | tuple[str, str] = ("Not SDG", "SDG"),
) -> str:
    return classification_report(y_true, y_pred, target_names=list(target_names))

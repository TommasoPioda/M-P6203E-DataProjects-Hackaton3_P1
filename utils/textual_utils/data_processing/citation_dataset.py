from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from src.data.data_utils import build_vector_text_columns
from src.features.feature_extractor import FeatureExtractor


class CitationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_extractor: FeatureExtractor | None = None, include_authors: bool = True):
        self.df = build_vector_text_columns(df.reset_index(drop=True), include_authors=include_authors)
        self.feature_extractor = feature_extractor or FeatureExtractor()

        if feature_extractor is None:
            self.feature_extractor.fit(
                self.df["vector_text_article"].tolist(),
                self.df["vector_text_ref"].tolist(),
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        article_features = self.feature_extractor.extract_features(row["vector_text_article"])
        ref_features = self.feature_extractor.extract_features(row["vector_text_ref"])

        return {
            "article_features": torch.tensor(article_features, dtype=torch.float32),
            "ref_features": torch.tensor(ref_features, dtype=torch.float32),
            "labels": torch.tensor(row["is_reference_valid"], dtype=torch.float32),
        }


class BertCitationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128, include_authors: bool = True):
        self.df = build_vector_text_columns(df.reset_index(drop=True), include_authors=include_authors)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        tokenized = self.tokenizer(
            row["vector_text_article"],
            row["vector_text_ref"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in tokenized.items()}
        item["labels"] = torch.tensor(row["is_reference_valid"], dtype=torch.float32)
        return item

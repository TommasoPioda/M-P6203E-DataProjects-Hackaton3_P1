import torch
from torch.utils.data import Dataset
import pandas as pd
from feature_extractor import FeatureExtractor

class CitationDataset(Dataset):
    """
    PyTorch Dataset for handling citation data.

    This class takes a DataFrame containing article and reference information,
    extracts features using a FeatureExtractor, and prepares the data for
    training a model.
    """
    def __init__(self, df: pd.DataFrame, feature_extractor: FeatureExtractor | None = None):
        """
        Initializes the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing the citation data. 
                               It must include columns for article and reference text,
                               and a label 'is_reference_valid'.
            feature_extractor (FeatureExtractor | None): Optional pre-fitted
                               feature extractor from feature_extractor.py.
                               If not provided, a new one is created and fitted
                               on the current dataframe.
        """
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor or FeatureExtractor()

        if feature_extractor is None:
            article_texts = self.df.apply(self._build_article_text, axis=1).tolist()
            ref_texts = self.df.apply(self._build_ref_text, axis=1).tolist()
            self.feature_extractor.fit(article_texts, ref_texts)

    @staticmethod
    def _build_article_text(row: pd.Series) -> str:
        """Build article text from dataframe columns."""
        return (
            f"{row.get('title_article', '')} "
            f"{row.get('abstract_article', '')} "
            f"{row.get('keywords_article', '')}"
        ).strip()

    @staticmethod
    def _build_ref_text(row: pd.Series) -> str:
        """Build reference text from dataframe columns."""
        return (
            f"{row.get('title_ref', '')} "
            f"{row.get('abstract_ref', '')} "
            f"{row.get('keywords_ref', '')}"
        ).strip()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the feature tensors and the corresponding label.
                  - 'article_features': Tensor of features for the article.
                  - 'ref_features': Tensor of features for the reference.
                  - 'labels': The ground truth label (1.0 for valid, 0.0 for invalid).
        """
        row = self.df.iloc[idx]

        article_text = self._build_article_text(row)
        ref_text = self._build_ref_text(row)

        # Extract numerical features from the text
        article_features = self.feature_extractor.extract_features(article_text)
        ref_features = self.feature_extractor.extract_features(ref_text)
        
        # Prepare the item for the model
        item = {
            'article_features': torch.tensor(article_features, dtype=torch.float),
            'ref_features': torch.tensor(ref_features, dtype=torch.float),
            'labels': torch.tensor(row['is_reference_valid'], dtype=torch.float)
        }

        return item


class BertCitationDataset(Dataset):
    """Dataset that tokenizes article-reference text pairs for BERT models."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    @staticmethod
    def _build_article_text(row: pd.Series) -> str:
        return (
            f"{row.get('title_article', '')} "
            f"{row.get('abstract_article', '')} "
            f"{row.get('keywords_article', '')}"
        ).strip()

    @staticmethod
    def _build_ref_text(row: pd.Series) -> str:
        return (
            f"{row.get('title_ref', '')} "
            f"{row.get('abstract_ref', '')} "
            f"{row.get('keywords_ref', '')}"
        ).strip()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        article_text = self._build_article_text(row)
        ref_text = self._build_ref_text(row)

        tokenized = self.tokenizer(
            article_text,
            ref_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in tokenized.items()}
        item["labels"] = torch.tensor(row["is_reference_valid"], dtype=torch.float)
        return item
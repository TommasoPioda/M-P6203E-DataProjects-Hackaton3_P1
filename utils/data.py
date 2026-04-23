from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def clean_references(value) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return []


def is_not_empty(value) -> bool:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return len(value) > 0
    if isinstance(value, dict):
        return any(str(item).strip() for item in value.values())
    if pd.isna(value):
        return False
    return len(str(value).strip()) > 0


def extract_author_names(authors) -> str:
    if isinstance(authors, dict):
        return str(authors.get("name", "")).strip()
    if isinstance(authors, (list, tuple, np.ndarray, pd.Series)):
        names: list[str] = []
        for item in authors:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
            else:
                name = str(item).strip()
            if name:
                names.append(name)
        return ", ".join(names)
    return ""


def normalize_text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def load_parquet_chunks(
    path_template: str | Path,
    parts: Iterable[int] | None = None,
    start: int = 1,
    end: int | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    template = str(path_template)
    if parts is None:
        if end is None:
            raise ValueError("Provide either parts or end when loading parquet chunks.")
        parts = range(start, end + 1)

    frames = [pd.read_parquet(template.format(part), columns=columns) for part in parts]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_parquet_files(
    file_paths: Sequence[str | Path],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frames = [pd.read_parquet(str(file_path), columns=columns) for file_path in file_paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_clean_citation_dataframe(
    path_template: str | Path,
    parts: Iterable[int] | None = None,
    start: int = 1,
    end: int | None = None,
    drop_columns: Sequence[str] = ("issn", "isbn"),
    remove_empty_rows: bool = True,
) -> pd.DataFrame:
    df = load_parquet_chunks(path_template, parts=parts, start=start, end=end)
    if df.empty:
        return df

    df = df.drop(columns=list(drop_columns), errors="ignore")
    df = df.dropna(subset=["id", "references"]).copy()
    df["id"] = df["id"].astype(str)
    df["references"] = df["references"].apply(clean_references)
    df["references"] = df["references"].apply(lambda values: [str(item) for item in values])
    df = df[df["references"].map(len) > 0].reset_index(drop=True)

    if remove_empty_rows:
        mask = df.apply(lambda row: all(is_not_empty(value) for value in row), axis=1)
        df = df[mask].reset_index(drop=True)

    return df


def load_clean_citation_dataframe_from_files(
    file_paths: Sequence[str | Path],
    drop_columns: Sequence[str] = ("issn", "isbn"),
    remove_empty_rows: bool = True,
) -> pd.DataFrame:
    df = load_parquet_files(file_paths)
    if df.empty:
        return df

    df = df.drop(columns=list(drop_columns), errors="ignore")
    df = df.dropna(subset=["id", "references"]).copy()
    df["id"] = df["id"].astype(str)
    df["references"] = df["references"].apply(clean_references)
    df["references"] = df["references"].apply(lambda values: [str(item) for item in values])
    df = df[df["references"].map(len) > 0].reset_index(drop=True)

    if remove_empty_rows:
        mask = df.apply(lambda row: all(is_not_empty(value) for value in row), axis=1)
        df = df[mask].reset_index(drop=True)

    return df


def build_positive_negative_pairs(
    df: pd.DataFrame,
    seed: int = 42,
    include_self_negatives: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    working = df.copy()
    working["id"] = working["id"].astype(str)
    working["references"] = working["references"].apply(lambda values: [str(item) for item in values])
    pairs = working[["id", "references"]].explode("references")
    pairs = pairs.rename(columns={"id": "article_id", "references": "ref_id"})
    pairs = pairs.dropna(subset=["ref_id"])
    pairs = pairs[pairs["ref_id"].astype(str).str.strip() != ""].reset_index(drop=True)

    ref_ids_in_dataset = set(working["id"].astype(str).values)
    pairs["ref_id"] = pairs["ref_id"].astype(str)
    pairs["article_id"] = pairs["article_id"].astype(str)
    pairs = pairs[pairs["ref_id"].isin(ref_ids_in_dataset)].reset_index(drop=True)

    article_features = working.drop(columns=["references"]).add_suffix("_article")
    article_features = article_features.rename(columns={"id_article": "article_id"})

    ref_features = working.drop(columns=["references"]).add_suffix("_ref")
    ref_features = ref_features.rename(columns={"id_ref": "ref_id"})

    df_pos = (
        pairs.merge(article_features, on="article_id", how="left")
        .merge(ref_features, on="ref_id", how="inner")
    )
    df_pos["is_reference_valid"] = 1

    rng = np.random.default_rng(seed)
    all_ids = working["id"].astype(str).to_numpy()
    refs_by_article = working.assign(id=working["id"].astype(str)).set_index("id")["references"].to_dict()

    fake_ref_ids: list[str] = []
    for article_id in pairs["article_id"].to_numpy():
        true_refs = {str(item) for item in refs_by_article.get(article_id, [])}
        if include_self_negatives:
            forbidden = true_refs
        else:
            forbidden = true_refs | {article_id}

        candidates = [candidate for candidate in all_ids if candidate not in forbidden]
        if not candidates:
            candidates = [candidate for candidate in all_ids if candidate not in true_refs]
        if not candidates:
            candidates = list(all_ids)
        fake_ref_ids.append(rng.choice(candidates))

    neg_pairs = pairs[["article_id"]].copy()
    neg_pairs["ref_id"] = fake_ref_ids

    df_neg = (
        neg_pairs.merge(article_features, on="article_id", how="left")
        .merge(ref_features, on="ref_id", how="inner")
    )
    df_neg["is_reference_valid"] = 0

    return df_pos, df_neg


def build_training_dataframe(
    df: pd.DataFrame,
    seed: int = 42,
    filter_years: bool = True,
) -> pd.DataFrame:
    df_pos, df_neg = build_positive_negative_pairs(df, seed=seed)
    if df_pos.empty and df_neg.empty:
        return pd.DataFrame()

    df_training = pd.concat([df_pos, df_neg], ignore_index=True)
    df_training = df_training.dropna()

    if filter_years and {"year_article", "year_ref"}.issubset(df_training.columns):
        df_training = df_training[df_training["year_article"] >= df_training["year_ref"]]

    if "year_article" in df_training.columns:
        df_training = df_training.sort_values("year_article")

    return df_training.reset_index(drop=True)


def build_vector_text_columns(df: pd.DataFrame, include_authors: bool = True) -> pd.DataFrame:
    working = df.copy()

    def build_article_text(row: pd.Series) -> str:
        parts = [
            normalize_text(row.get("title_article", "")),
            normalize_text(row.get("abstract_article", "")),
            normalize_text(row.get("keywords_article", "")),
        ]
        if include_authors:
            parts.append(extract_author_names(row.get("authors_article", "")))
        return " ".join(part for part in parts if part).strip()

    def build_ref_text(row: pd.Series) -> str:
        parts = [
            normalize_text(row.get("title_ref", "")),
            normalize_text(row.get("abstract_ref", "")),
            normalize_text(row.get("keywords_ref", "")),
        ]
        if include_authors:
            parts.append(extract_author_names(row.get("authors_ref", "")))
        return " ".join(part for part in parts if part).strip()

    working["vector_text_article"] = working.apply(build_article_text, axis=1).fillna("").astype(str)
    working["vector_text_ref"] = working.apply(build_ref_text, axis=1).fillna("").astype(str)
    return working


def resolve_chunk_paths(
    path_template: str | Path,
    parts: Iterable[int] | None = None,
    start: int = 1,
    end: int | None = None,
) -> list[str]:
    if parts is None:
        if end is None:
            raise ValueError("Provide either parts or end when resolving chunk paths.")
        parts = range(start, end + 1)
    template = str(path_template)
    return [template.format(part) for part in parts]


def chunk_sequence(items: Sequence, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]

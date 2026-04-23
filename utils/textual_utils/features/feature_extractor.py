from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Use sklearn for stability with list inputs
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

USE_GPU = False


@dataclass
class FeatureExtractorConfig:
    max_features: int = 10000
    stop_words: str | None = "english"
    ngram_range: tuple[int, int] = (1, 2)


class FeatureExtractor:
    def __init__(self, max_features: int = 10000, stop_words: str | None = "english", ngram_range: tuple[int, int] = (1, 2)):
        self.config = FeatureExtractorConfig(
            max_features=max_features,
            stop_words=stop_words,
            ngram_range=ngram_range,
        )
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            stop_words=self.config.stop_words,
            ngram_range=self.config.ngram_range,
            
        )
        self.article_reducer: TruncatedSVD | None = None
        self.ref_reducer: TruncatedSVD | None = None

    def fit(self, articles: list[str], refs: list[str]) -> None:
        # Combine articles and references into a single list for vocabulary fitting
        combined = list(articles) + list(refs)
        self.vectorizer.fit(combined)

    def transform(self, articles: list[str], refs: list[str]) -> tuple:
        tfidf_articles = self.vectorizer.transform(articles)
        tfidf_refs = self.vectorizer.transform(refs)

        if tfidf_articles.shape[0] != tfidf_refs.shape[0]:
            raise ValueError("articles and refs must have the same number of rows")

        dot_products = np.asarray(tfidf_articles.multiply(tfidf_refs).sum(axis=1)).ravel()
        norm_articles = np.sqrt(np.asarray(tfidf_articles.multiply(tfidf_articles).sum(axis=1)).ravel())
        norm_refs = np.sqrt(np.asarray(tfidf_refs.multiply(tfidf_refs).sum(axis=1)).ravel())
        denominators = norm_articles * norm_refs

        sims = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=denominators > 0,
        )

        return tfidf_articles, tfidf_refs, sims.tolist()

    def extract_features(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text]).toarray().flatten()

    def fit_reducers(
        self,
        tfidf_articles,
        tfidf_refs,
        n_components: int = 256,
        random_state: int = 42,
    ) -> None:
        if tfidf_articles.shape[1] <= n_components or tfidf_refs.shape[1] <= n_components:
            raise ValueError(
                "n_components must be smaller than the number of TF-IDF features. "
                f"Got n_components={n_components}, article_features={tfidf_articles.shape[1]}, ref_features={tfidf_refs.shape[1]}"
            )

        self.article_reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.ref_reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        
        self.article_reducer.fit(tfidf_articles)
        self.ref_reducer.fit(tfidf_refs)

    def transform_reduced(self, tfidf_articles, tfidf_refs) -> tuple[np.ndarray, np.ndarray]:
        if self.article_reducer is None or self.ref_reducer is None:
            raise RuntimeError("Reducers are not fitted. Call fit_reducers(...) first.")

        article_embeddings = self.article_reducer.transform(tfidf_articles).astype(np.float32)
        ref_embeddings = self.ref_reducer.transform(tfidf_refs).astype(np.float32)
        return article_embeddings, ref_embeddings

    def build_flat_embeddings_dataframe(
        self,
        base_df: pd.DataFrame,
        tfidf_articles,
        tfidf_refs,
        fit_tfidf_articles=None,
        fit_tfidf_refs=None,
        n_components: int = 256,
        random_state: int = 42,
        meta_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        meta_columns = meta_columns or ["split", "article_id", "ref_id", "is_reference_valid"]
        missing_columns = [column for column in meta_columns if column not in base_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required metadata columns: {missing_columns}")

        self.fit_reducers(
            tfidf_articles=fit_tfidf_articles if fit_tfidf_articles is not None else tfidf_articles,
            tfidf_refs=fit_tfidf_refs if fit_tfidf_refs is not None else tfidf_refs,
            n_components=n_components,
            random_state=random_state,
        )
        article_embeddings, ref_embeddings = self.transform_reduced(tfidf_articles, tfidf_refs)

        article_cols = [f"article_emb_{i:03d}" for i in range(article_embeddings.shape[1])]
        ref_cols = [f"ref_emb_{i:03d}" for i in range(ref_embeddings.shape[1])]

        df_article_emb = pd.DataFrame(article_embeddings, columns=article_cols, index=base_df.index)
        df_ref_emb = pd.DataFrame(ref_embeddings, columns=ref_cols, index=base_df.index)

        return pd.concat(
            [
                base_df[meta_columns].reset_index(drop=True),
                df_article_emb.reset_index(drop=True),
                df_ref_emb.reset_index(drop=True),
            ],
            axis=1,
        )


def build_classic_ml_matrix(
    df: "pd.DataFrame",
    max_features: int = 256,
    include_authors: bool = True,
) -> tuple:
    import pandas as pd
    import scipy.sparse as sp

    from src.data.data_utils import build_vector_text_columns

    working = build_vector_text_columns(df, include_authors=include_authors)
    y = working["is_reference_valid"].to_numpy()
    X = working.drop(columns=["is_reference_valid", "article_id", "ref_id"], errors="ignore").copy()

    articles = X.pop("vector_text_article").fillna("").astype(str).tolist()
    refs = X.pop("vector_text_ref").fillna("").astype(str).tolist()

    drop_non_numeric_cols = [
        "title_article",
        "abstract_article",
        "keywords_article",
        "lang_article",
        "authors_article",
        "venue_article",
        "doc_type_article",
        "doi_article",
        "url_article",
        "title_ref",
        "abstract_ref",
        "keywords_ref",
        "lang_ref",
        "authors_ref",
        "venue_ref",
        "doc_type_ref",
        "doi_ref",
        "url_ref",
    ]

    meta = X.drop(columns=drop_non_numeric_cols, errors="ignore").copy()
    for column in meta.columns:
        meta[column] = pd.to_numeric(meta[column], errors="coerce").fillna(0.0)

    feature_extractor = FeatureExtractor(max_features=max_features)
    feature_extractor.fit(articles, refs)
    X_articles, X_refs, sims = feature_extractor.transform(articles, refs)

    if meta.shape[1] > 0:
        X_meta = sp.csr_matrix(meta.to_numpy(dtype=np.float32))
    else:
        X_meta = sp.csr_matrix((len(working), 0), dtype=np.float32)

    sims_col = sp.csr_matrix(np.asarray(sims, dtype=np.float32).reshape(-1, 1))
    X_model = sp.hstack([X_meta, X_articles.tocsr(), X_refs.tocsr(), sims_col], format="csr")

    return X_model, y, feature_extractor, {
        "working_df": working,
        "meta_columns": list(meta.columns),
    }

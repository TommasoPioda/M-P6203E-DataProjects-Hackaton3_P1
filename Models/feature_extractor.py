from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class FeatureExtractor:
    """
    A class to extract TF-IDF features and compute cosine similarity between text pairs.
    """
    def __init__(self, max_features: int = 10000):
        """
        Initializes the TfidfVectorizer.

        Args:
            max_features (int): The maximum number of features (words) to consider.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Consider both unigrams and bigrams
        )

    def fit(self, articles: list[str], refs: list[str]):
        """
        Fits the TF-IDF vectorizer on the combined corpus of articles and references.

        Args:
            articles (list[str]): A list of article texts.
            refs (list[str]): A list of reference texts.
        """
        # Combine articles and references to build a comprehensive vocabulary
        combined_texts = articles + refs
        self.vectorizer.fit(combined_texts)

    def transform(self, articles: list[str], refs: list[str]) -> tuple:
        """
        Transforms the articles and references into TF-IDF vectors and calculates
        the cosine similarity between each pair.

        Args:
            articles (list[str]): A list of article texts.
            refs (list[str]): A list of reference texts.

        Returns:
            tuple: A tuple containing:
                - tfidf_articles: Sparse matrix of TF-IDF features for articles.
                - tfidf_refs: Sparse matrix of TF-IDF features for references.
                - sims: A list of cosine similarity scores between each article-reference pair.
        """
        # Transform articles and references into TF-IDF vectors
        tfidf_articles = self.vectorizer.transform(articles)
        tfidf_refs = self.vectorizer.transform(refs)

        # Calculate cosine similarity for each corresponding article-reference pair
        # without materializing an NxN similarity matrix.
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
        """
        Extracts TF-IDF features for a single piece of text.
        Note: This method is intended for use after the vectorizer has been fitted.

        Args:
            text (str): The text to transform.

        Returns:
            np.ndarray: A dense numpy array of TF-IDF features.
        """
        return self.vectorizer.transform([text]).toarray().flatten()

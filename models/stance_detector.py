"""
stance_detector.py — TF-IDF + Logistic Regression Stance Detector for RNIA
===========================================================================

Trains and uses a TF-IDF + Logistic Regression pipeline to detect the
sentiment stance of a news article.

Stance Labels:
    positive, negative, neutral

Usage:
    # Training
    >>> from models.stance_detector import StanceDetectorLR
    >>> det = StanceDetectorLR()
    >>> det.train_stance_model(X_train, y_train)
    >>> det.save_model()

    # Prediction
    >>> det = StanceDetectorLR()
    >>> det.load_model()
    >>> stance = det.predict_stance("Company files for bankruptcy")
    >>> print(stance)
    negative
"""

import os
import sys
import re
import logging

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths for saved models
# ---------------------------------------------------------------------------
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
STANCE_CLF_PATH = os.path.join(SAVED_MODELS_DIR, "stance_detector.pkl")
TFIDF_STANCE_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_stance_vectorizer.pkl")

# Valid stance labels (V2 market-oriented, 3-class — mixed collapsed into neutral)
STANCE_LABELS = ["bullish", "bearish", "neutral"]


class StanceDetectorLR:
    """
    Stance detector using TF-IDF vectorization + Logistic Regression.

    Classifies article text as positive, negative, or neutral.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer for transforming text to feature vectors.
    classifier : LogisticRegression
        Trained logistic regression classifier for stance labels.
    is_trained : bool
        Whether the model has been trained or loaded.
    """

    # Minimum confidence to trust a prediction (else → "neutral")
    CONFIDENCE_THRESHOLD = 0.45
    # Minimum word count for meaningful stance detection
    MIN_WORDS = 5

    def __init__(self):
        """Initialise the stance detector with default hyperparameters."""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,      # Larger vocabulary for better coverage
            ngram_range=(1, 2),      # Unigrams and bigrams
            stop_words="english",    # Remove common English stop words
            min_df=2,                # Require term in at least 2 docs (noise filter)
            max_df=0.95,             # Ignore terms in >95% of docs (too generic)
            sublinear_tf=True,       # Apply sublinear TF scaling
        )
        self.classifier = LogisticRegression(
            max_iter=2000,           # Ensure convergence
            solver="lbfgs",
            C=1.0,                   # Balanced regularisation
            class_weight="balanced", # Handle class imbalance automatically
            random_state=42,
        )
        self.is_trained = False
        logger.info("StanceDetectorLR initialised.")

    # ----- Training ---------------------------------------------------------

    def train_stance_model(self, X_train: list[str], y_train: list[str]) -> None:
        """
        Train the stance detection model.

        Parameters
        ----------
        X_train : list[str]
            List of cleaned article texts for training.
        y_train : list[str]
            Corresponding stance labels (``"positive"``, ``"negative"``,
            or ``"neutral"``).

        Returns
        -------
        None
        """
        logger.info("Training stance detector on %d samples...", len(X_train))

        # Step 1 — Fit TF-IDF vectorizer and transform training text
        X_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info("  TF-IDF matrix shape: %s", X_tfidf.shape)

        # Step 2 — Train Logistic Regression classifier
        self.classifier.fit(X_tfidf, y_train)
        self.is_trained = True

        # Training accuracy
        train_acc = self.classifier.score(X_tfidf, y_train)
        logger.info("  Training accuracy: %.4f", train_acc)
        logger.info("Stance detector training complete.")

    # ----- Prediction -------------------------------------------------------

    @staticmethod
    def _is_meaningful_text(text: str, min_words: int = 5) -> bool:
        """Return True if *text* contains enough real words for classification."""
        if not text or not isinstance(text, str):
            return False
        words = re.findall(r"[a-zA-Z]{2,}", text)
        return len(words) >= min_words

    def predict_stance(self, text: str, strict: bool = False) -> str:
        """
        Predict the stance for a single article text.

        Parameters
        ----------
        text : str
            Cleaned article text.
        strict : bool
            If True, apply input-quality and confidence checks —
            returns ``"neutral"`` for gibberish or low-confidence
            inputs.  Use ``strict=True`` for live user input via the API.
            Default is False (pipeline / evaluation mode).

        Returns
        -------
        str
            Predicted stance label (``"positive"``, ``"negative"``,
            or ``"neutral"``).
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model not trained. Call train_stance_model() or load_model() first."
            )

        if strict:
            # Reject gibberish / too-short input
            if not self._is_meaningful_text(text, self.MIN_WORDS):
                return "neutral"

        X_tfidf = self.vectorizer.transform([text])

        if strict:
            # Check if the text produced any known TF-IDF features
            if X_tfidf.nnz == 0:
                return "neutral"

            # Confidence check
            probas = self.classifier.predict_proba(X_tfidf)[0]
            max_prob = float(np.max(probas))
            if max_prob < self.CONFIDENCE_THRESHOLD:
                return "neutral"

        prediction = self.classifier.predict(X_tfidf)
        return prediction[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Get prediction probabilities for all stance labels.

        Parameters
        ----------
        text : str
            Cleaned article text.

        Returns
        -------
        dict[str, float]
            Mapping of stance label → probability.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X_tfidf = self.vectorizer.transform([text])
        probas = self.classifier.predict_proba(X_tfidf)[0]
        classes = self.classifier.classes_
        return {cls: round(prob, 4) for cls, prob in zip(classes, probas)}

    # ----- Save / Load -------------------------------------------------------

    def save_model(
        self,
        clf_path: str = STANCE_CLF_PATH,
        vec_path: str = TFIDF_STANCE_PATH,
    ) -> None:
        """
        Save the trained model and vectorizer to disk using joblib.

        Parameters
        ----------
        clf_path : str
            File path for the stance classifier pickle.
        vec_path : str
            File path for the TF-IDF vectorizer pickle.

        Notes
        -----
        If the event classifier and stance detector share the same TF-IDF
        vectorizer, the vectorizer file will be overwritten with the stance
        detector's version. Use ``train_models.py`` for coordinated saving.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not trained.")

        os.makedirs(os.path.dirname(clf_path), exist_ok=True)

        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.vectorizer, vec_path)

        logger.info("Stance detector saved to: %s", clf_path)
        logger.info("TF-IDF vectorizer saved to: %s", vec_path)

    def load_model(
        self,
        clf_path: str = STANCE_CLF_PATH,
        vec_path: str = TFIDF_STANCE_PATH,
    ) -> None:
        """
        Load a previously trained model and vectorizer from disk.

        Parameters
        ----------
        clf_path : str
            File path for the stance classifier pickle.
        vec_path : str
            File path for the TF-IDF vectorizer pickle.
        """
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Stance classifier not found: {clf_path}")
        if not os.path.isfile(vec_path):
            raise FileNotFoundError(f"Vectorizer not found: {vec_path}")

        self.classifier = joblib.load(clf_path)
        self.vectorizer = joblib.load(vec_path)
        self.is_trained = True

        logger.info("Stance detector loaded from: %s", clf_path)
        logger.info("TF-IDF vectorizer loaded from: %s", vec_path)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Demo with small synthetic data
    sample_texts = [
        "company reports record profits and raises dividend by twenty percent",
        "massive layoffs announced as revenue plummets",
        "fed holds interest rates steady as widely expected",
        "new product launch generates excitement among investors",
        "stock plunges after disappointing earnings report",
        "quarterly results in line with analyst expectations",
    ]
    sample_labels = [
        "positive", "negative", "neutral",
        "positive", "negative", "neutral",
    ]

    det = StanceDetectorLR()
    det.train_stance_model(sample_texts, sample_labels)

    print("\nStance Prediction Test:")
    print("-" * 50)
    test = "company announces massive job cuts amid declining sales"
    pred = det.predict_stance(test)
    print(f"  Input : {test}")
    print(f"  Pred  : {pred}")

"""
event_classifier.py — TF-IDF + Logistic Regression Event Classifier for RNIA
=============================================================================

Trains and uses a TF-IDF + Logistic Regression pipeline to classify news
articles into one of seven financial event categories.

Event Categories (7 — consolidated):
    1. Earnings
    2. Leadership_Change
    3. Regulatory_Action
    4. Mergers_Acquisitions
    5. Legal_Action
    6. Product_Announcement
    7. Market_Movement  (includes Macroeconomic, Market Sentiment, Other)

Usage:
    # Training
    >>> from models.event_classifier import EventClassifierLR
    >>> clf = EventClassifierLR()
    >>> clf.train(X_train, y_train)
    >>> clf.save_model()

    # Prediction
    >>> clf = EventClassifierLR()
    >>> clf.load_model()
    >>> event_type = clf.predict("Apple reports record quarterly earnings")
    >>> print(event_type)
    earnings
"""

import os
import sys
import re
import logging

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from taxonomy.event_taxonomy import EVENT_CATEGORIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths for saved models
# ---------------------------------------------------------------------------
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
EVENT_CLF_PATH = os.path.join(SAVED_MODELS_DIR, "event_classifier.pkl")
TFIDF_EVENT_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_vectorizer.pkl")


class EventClassifierLR:
    """
    Event classifier using TF-IDF vectorization + Logistic Regression.

    This classifier converts article text into TF-IDF feature vectors and
    uses a Logistic Regression model to predict the event category.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer for transforming text to feature vectors.
    classifier : LogisticRegression
        Trained logistic regression classifier.
    is_trained : bool
        Whether the model has been trained or loaded.
    """

    # Minimum confidence to trust a prediction (else → "unclassified")
    CONFIDENCE_THRESHOLD = 0.35
    # Minimum word count for meaningful classification
    MIN_WORDS = 5

    def __init__(self):
        """Initialise the event classifier with default hyperparameters."""
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
        logger.info("EventClassifierLR initialised.")

    # ----- Training ---------------------------------------------------------

    def train_event_classifier(self, X_train: list[str], y_train: list[str]) -> None:
        """
        Train the event classification model.

        Parameters
        ----------
        X_train : list[str]
            List of cleaned article texts for training.
        y_train : list[str]
            Corresponding event-type labels (e.g. ``"earnings"``).

        Returns
        -------
        None
        """
        logger.info("Training event classifier on %d samples...", len(X_train))

        # Step 1 — Fit TF-IDF vectorizer and transform training text
        X_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info("  TF-IDF matrix shape: %s", X_tfidf.shape)

        # Step 2 — Train Logistic Regression classifier
        self.classifier.fit(X_tfidf, y_train)
        self.is_trained = True

        # Training accuracy
        train_acc = self.classifier.score(X_tfidf, y_train)
        logger.info("  Training accuracy: %.4f", train_acc)
        logger.info("Event classifier training complete.")

    # ----- Prediction -------------------------------------------------------

    @staticmethod
    def _is_meaningful_text(text: str, min_words: int = 5) -> bool:
        """Return True if *text* contains enough real words for classification."""
        if not text or not isinstance(text, str):
            return False
        words = re.findall(r"[a-zA-Z]{2,}", text)
        return len(words) >= min_words

    def predict_event(self, text: str, strict: bool = False) -> str:
        """
        Predict the event category for a single article text.

        Parameters
        ----------
        text : str
            Cleaned article text.
        strict : bool
            If True, apply input-quality and confidence checks —
            returns ``"unclassified"`` for gibberish or low-confidence
            inputs.  Use ``strict=True`` for live user input via the API.
            Default is False (pipeline / evaluation mode).

        Returns
        -------
        str
            Predicted event category label, or ``"unclassified"`` when
            *strict* is True and the input fails validation.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model not trained. Call train_event_classifier() or load_model() first."
            )

        if strict:
            # Reject gibberish / too-short input
            if not self._is_meaningful_text(text, self.MIN_WORDS):
                return "unclassified"

        X_tfidf = self.vectorizer.transform([text])

        if strict:
            # Check if the text produced any known TF-IDF features
            if X_tfidf.nnz == 0:
                return "unclassified"

            # Confidence check
            probas = self.classifier.predict_proba(X_tfidf)[0]
            max_prob = float(np.max(probas))
            if max_prob < self.CONFIDENCE_THRESHOLD:
                return "unclassified"

        prediction = self.classifier.predict(X_tfidf)
        return prediction[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        """
        Get prediction probabilities for all event categories.

        Parameters
        ----------
        text : str
            Cleaned article text.

        Returns
        -------
        dict[str, float]
            Mapping of event category → probability.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X_tfidf = self.vectorizer.transform([text])
        probas = self.classifier.predict_proba(X_tfidf)[0]
        classes = self.classifier.classes_
        return {str(cls): float(f"{float(prob):.4f}") for cls, prob in zip(classes, probas)}

    # ----- Save / Load -------------------------------------------------------

    def save_model(self, clf_path: str = EVENT_CLF_PATH, vec_path: str = TFIDF_EVENT_PATH) -> None:
        """
        Save the trained model and vectorizer to disk using joblib.

        Parameters
        ----------
        clf_path : str
            File path for the classifier pickle.
        vec_path : str
            File path for the TF-IDF vectorizer pickle.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not trained.")

        os.makedirs(os.path.dirname(clf_path), exist_ok=True)

        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.vectorizer, vec_path)

        logger.info("Event classifier saved to: %s", clf_path)
        logger.info("TF-IDF vectorizer saved to: %s", vec_path)

    def load_model(self, clf_path: str = EVENT_CLF_PATH, vec_path: str = TFIDF_EVENT_PATH) -> None:
        """
        Load a previously trained model and vectorizer from disk.

        Parameters
        ----------
        clf_path : str
            File path for the classifier pickle.
        vec_path : str
            File path for the TF-IDF vectorizer pickle.
        """
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Classifier not found: {clf_path}")
        if not os.path.isfile(vec_path):
            raise FileNotFoundError(f"Vectorizer not found: {vec_path}")

        self.classifier = joblib.load(clf_path)
        self.vectorizer = joblib.load(vec_path)
        self.is_trained = True

        logger.info("Event classifier loaded from: %s", clf_path)
        logger.info("TF-IDF vectorizer loaded from: %s", vec_path)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    # Demo with small synthetic data
    sample_texts = [
        "company reports strong revenue growth and record profits this quarter",
        "ceo announces resignation amid board restructuring",
        "government introduces new regulations on data privacy",
        "two major firms announce merger deal worth billions",
        "company files lawsuit against competitor for patent violation",
        "tech giant launches new smartphone with ai features",
        "stock market falls sharply after interest rate hike announcement",
    ]
    sample_labels = [
        "earnings", "leadership_change", "regulatory_action",
        "mergers_acquisitions", "legal_action", "product_announcement",
        "market_movement",
    ]

    clf = EventClassifierLR()
    clf.train_event_classifier(sample_texts, sample_labels)

    print("\nPrediction Test:")
    print("-" * 50)
    test = "apple posted record quarterly earnings beating expectations"
    pred = clf.predict_event(test)
    print(f"  Input : {test}")
    print(f"  Pred  : {pred}")

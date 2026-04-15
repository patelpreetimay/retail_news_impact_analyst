"""
impact_regressor.py — Content-Aware Impact Score Regressor for RNIA
====================================================================

Trains and uses a TF-IDF + Ridge Regression model to predict the
impact score (0.0–1.0) directly from article text.

Unlike the V1 heuristic scorer (which only looked at source name,
timestamp, and event type), this model reads the article content and
produces a score that captures what V2's Gemini Flash + Pro pipeline
would have assigned.

Usage:
    # Training
    >>> from models.impact_regressor import ImpactRegressor
    >>> reg = ImpactRegressor()
    >>> reg.train(X_train_texts, y_train_scores)
    >>> reg.save_model()

    # Prediction
    >>> reg = ImpactRegressor()
    >>> reg.load_model()
    >>> score = reg.predict("Apple reports record quarterly earnings")
    >>> print(score)
    0.87
"""

import os
import sys
import logging

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

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
IMPACT_REG_PATH = os.path.join(SAVED_MODELS_DIR, "impact_regressor.pkl")
TFIDF_IMPACT_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_impact_vectorizer.pkl")


class ImpactRegressor:
    """
    Content-aware impact score regressor using TF-IDF + Ridge Regression.

    Predicts a continuous impact score (0.0–1.0) from article text,
    trained on V2's Gemini-validated impact scores.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer for transforming text to feature vectors.
    regressor : Ridge
        Trained Ridge regression model.
    is_trained : bool
        Whether the model has been trained or loaded.
    """

    def __init__(self):
        """Initialise the impact regressor with default hyperparameters."""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.regressor = Ridge(
            alpha=1.0,
            random_state=42,
        )
        self.is_trained = False
        logger.info("ImpactRegressor initialised.")

    # ----- Training ---------------------------------------------------------

    def train(self, X_train: list[str], y_train: list[float]) -> None:
        """
        Train the impact regression model.

        Parameters
        ----------
        X_train : list[str]
            List of cleaned article texts for training.
        y_train : list[float]
            Corresponding impact scores (0.0–1.0).
        """
        logger.info("Training impact regressor on %d samples...", len(X_train))

        X_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info("  TF-IDF matrix shape: %s", X_tfidf.shape)

        self.regressor.fit(X_tfidf, y_train)
        self.is_trained = True

        # Training R²
        train_r2 = self.regressor.score(X_tfidf, y_train)
        logger.info("  Training R²: %.4f", train_r2)
        logger.info("Impact regressor training complete.")

    # ----- Prediction -------------------------------------------------------

    def predict(self, text: str) -> float:
        """
        Predict the impact score for a single article text.

        Parameters
        ----------
        text : str
            Cleaned article text.

        Returns
        -------
        float
            Predicted impact score, clamped to [0.0, 1.0].
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model not trained. Call train() or load_model() first."
            )

        X_tfidf = self.vectorizer.transform([text])
        raw_score = self.regressor.predict(X_tfidf)[0]

        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, raw_score)))

    def predict_batch(self, texts: list[str]) -> list[float]:
        """
        Predict impact scores for a batch of articles.

        Parameters
        ----------
        texts : list[str]
            List of cleaned article texts.

        Returns
        -------
        list[float]
            List of predicted impact scores, clamped to [0.0, 1.0].
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")

        X_tfidf = self.vectorizer.transform(texts)
        raw_scores = self.regressor.predict(X_tfidf)
        return [float(max(0.0, min(1.0, s))) for s in raw_scores]

    # ----- Save / Load -------------------------------------------------------

    def save_model(
        self,
        reg_path: str = IMPACT_REG_PATH,
        vec_path: str = TFIDF_IMPACT_PATH,
    ) -> None:
        """Save the trained model and vectorizer to disk using joblib."""
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not trained.")

        os.makedirs(os.path.dirname(reg_path), exist_ok=True)

        joblib.dump(self.regressor, reg_path)
        joblib.dump(self.vectorizer, vec_path)

        logger.info("Impact regressor saved to: %s", reg_path)
        logger.info("TF-IDF vectorizer saved to: %s", vec_path)

    def load_model(
        self,
        reg_path: str = IMPACT_REG_PATH,
        vec_path: str = TFIDF_IMPACT_PATH,
    ) -> None:
        """Load a previously trained model and vectorizer from disk."""
        if not os.path.isfile(reg_path):
            raise FileNotFoundError(f"Impact regressor not found: {reg_path}")
        if not os.path.isfile(vec_path):
            raise FileNotFoundError(f"Vectorizer not found: {vec_path}")

        self.regressor = joblib.load(reg_path)
        self.vectorizer = joblib.load(vec_path)
        self.is_trained = True

        logger.info("Impact regressor loaded from: %s", reg_path)
        logger.info("TF-IDF vectorizer loaded from: %s", vec_path)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # Demo with small synthetic data
    sample_texts = [
        "company reports record profits and raises dividend by twenty percent",
        "massive layoffs announced as revenue plummets sharply",
        "fed holds interest rates steady as widely expected",
        "new product launch generates excitement among investors and analysts",
        "stock plunges after disappointing earnings report misses estimates",
        "quarterly results in line with analyst expectations nothing special",
    ]
    sample_scores = [0.85, 0.92, 0.70, 0.65, 0.88, 0.55]

    reg = ImpactRegressor()
    reg.vectorizer.set_params(min_df=1)
    reg.train(sample_texts, sample_scores)

    print("\nImpact Prediction Test:")
    print("-" * 50)
    test = "apple posted record quarterly earnings beating expectations"
    pred = reg.predict(test)
    print(f"  Input : {test}")
    print(f"  Pred  : {pred:.4f}")

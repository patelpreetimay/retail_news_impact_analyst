"""
event_classifier_xgb.py — TF-IDF + XGBoost Event Classifier for RNIA
=====================================================================

Drop-in alternative to ``EventClassifierLR`` that uses gradient boosting
(XGBoost) instead of Logistic Regression on top of the same TF-IDF
features. Often beats LR by 3-6% F1 on sparse text features for
imbalanced multi-class problems like the 7-class event taxonomy.

Saved under DIFFERENT filenames so the LR baseline keeps working:
    models/saved_models/event_classifier_xgb.pkl
    models/saved_models/tfidf_event_xgb_vectorizer.pkl
    models/saved_models/event_xgb_label_encoder.pkl

Same public API as ``EventClassifierLR``:
    train_event_classifier(X, y), predict_event(text, strict=False),
    predict_proba(text), save_model(), load_model(), is_trained
"""

from __future__ import annotations

import logging
import os
import re
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
EVENT_XGB_PATH = os.path.join(SAVED_MODELS_DIR, "event_classifier_xgb.pkl")
TFIDF_EVENT_XGB_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_event_xgb_vectorizer.pkl")
EVENT_XGB_LABELENC_PATH = os.path.join(SAVED_MODELS_DIR, "event_xgb_label_encoder.pkl")


class EventClassifierXGB:
    """
    Event classifier using TF-IDF + XGBoost. Same shape as EventClassifierLR.

    Notes
    -----
    XGBoost requires integer class labels, so a LabelEncoder is fitted at
    train time and used to convert string labels ↔ integers.
    """

    CONFIDENCE_THRESHOLD = 0.35
    MIN_WORDS = 5

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Install with:  pip install xgboost"
            ) from e

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

        self.label_encoder = LabelEncoder()

        # XGBoost handles sparse matrices well via tree_method='hist'.
        # Disable use_label_encoder warning (we manage encoding ourselves).
        self.classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self.is_trained = False
        logger.info("EventClassifierXGB initialised "
                    "(n_estimators=%d, max_depth=%d, lr=%.2f).",
                    n_estimators, max_depth, learning_rate)

    # ----- Training ---------------------------------------------------------

    def train_event_classifier(self, X_train: list[str], y_train: list[str]) -> None:
        """Fit TF-IDF + XGBoost on training texts/labels."""
        logger.info("Training EventClassifierXGB on %d samples...", len(X_train))
        X_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info("  TF-IDF matrix shape: %s", X_tfidf.shape)

        y_int = self.label_encoder.fit_transform(y_train)
        logger.info("  Classes: %s", list(self.label_encoder.classes_))

        self.classifier.fit(X_tfidf, y_int)
        self.is_trained = True

        train_acc = self.classifier.score(X_tfidf, y_int)
        logger.info("  Training accuracy: %.4f", train_acc)
        logger.info("Event classifier (XGB) training complete.")

    # Compatibility alias — matches train_models.py style expected by callers
    def train(self, X_train: list[str], y_train: list[str]) -> None:
        return self.train_event_classifier(X_train, y_train)

    # ----- Prediction -------------------------------------------------------

    @staticmethod
    def _is_meaningful_text(text: str, min_words: int = 5) -> bool:
        if not text or not isinstance(text, str):
            return False
        words = re.findall(r"[a-zA-Z]{2,}", text)
        return len(words) >= min_words

    def predict_event(self, text: str, strict: bool = False) -> str:
        """Predict event label. Same semantics as EventClassifierLR.predict_event."""
        if not self.is_trained:
            raise RuntimeError(
                "Model not trained. Call train_event_classifier() or load_model() first."
            )

        if strict and not self._is_meaningful_text(text, self.MIN_WORDS):
            return "unclassified"

        X_tfidf = self.vectorizer.transform([text])

        if strict:
            if X_tfidf.nnz == 0:
                return "unclassified"
            probas = self.classifier.predict_proba(X_tfidf)[0]
            if float(np.max(probas)) < self.CONFIDENCE_THRESHOLD:
                return "unclassified"

        pred_int = int(self.classifier.predict(X_tfidf)[0])
        return self.label_encoder.inverse_transform([pred_int])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return {label: probability} dict — matches LR API."""
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X_tfidf = self.vectorizer.transform([text])
        probas = self.classifier.predict_proba(X_tfidf)[0]
        classes = self.label_encoder.classes_
        return {str(cls): float(f"{float(prob):.4f}") for cls, prob in zip(classes, probas)}

    # Mirror LR's classes_ surface for ensemble compatibility
    @property
    def classes_(self):
        if not self.is_trained:
            return np.array([])
        return np.asarray(self.label_encoder.classes_)

    # ----- Save / Load -------------------------------------------------------

    def save_model(
        self,
        clf_path: str = EVENT_XGB_PATH,
        vec_path: str = TFIDF_EVENT_XGB_PATH,
        label_path: str = EVENT_XGB_LABELENC_PATH,
    ) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not trained.")

        os.makedirs(os.path.dirname(clf_path), exist_ok=True)
        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.label_encoder, label_path)

        logger.info("EventClassifierXGB saved:")
        logger.info("  Classifier      : %s", clf_path)
        logger.info("  Vectorizer      : %s", vec_path)
        logger.info("  Label encoder   : %s", label_path)

    def load_model(
        self,
        clf_path: str = EVENT_XGB_PATH,
        vec_path: str = TFIDF_EVENT_XGB_PATH,
        label_path: str = EVENT_XGB_LABELENC_PATH,
    ) -> None:
        for p in (clf_path, vec_path, label_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing artifact: {p}")
        self.classifier = joblib.load(clf_path)
        self.vectorizer = joblib.load(vec_path)
        self.label_encoder = joblib.load(label_path)
        self.is_trained = True
        logger.info("EventClassifierXGB loaded from %s", clf_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
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
    clf = EventClassifierXGB(n_estimators=50, max_depth=3)
    clf.vectorizer.set_params(min_df=1, max_df=1.0)
    clf.train_event_classifier(sample_texts, sample_labels)
    print("\nXGB Prediction Test:")
    print(f"  -> {clf.predict_event('apple beat earnings expectations')}")

"""
event_classifier_hybrid.py — TF-IDF + FinBERT embeddings → Logistic Regression
================================================================================

Augments the TF-IDF feature space with 768-dim FinBERT [CLS] embeddings,
then trains a single Logistic Regression head on the concatenated
features. Best of both worlds:

    - Semantic understanding (from FinBERT contextual embeddings)
    - Interpretable per-feature coefficients (LR head, not a transformer)
    - Same lightweight inference contract: vector → LR.predict_proba

Saved under separate filenames so the LR-only baseline remains intact:
    models/saved_models/event_classifier_hybrid.pkl
    models/saved_models/tfidf_event_hybrid_vectorizer.pkl
    models/saved_models/event_hybrid_label_encoder.pkl

Same public API as ``EventClassifierLR`` so it can be swapped in
training, evaluation, and pipeline code with minimal change.
"""

from __future__ import annotations

import logging
import os
import re
import sys

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.finbert_embedder import FinBERTEmbedder

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
EVENT_HYBRID_PATH = os.path.join(SAVED_MODELS_DIR, "event_classifier_hybrid.pkl")
TFIDF_EVENT_HYBRID_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_event_hybrid_vectorizer.pkl")
EVENT_HYBRID_LABELENC_PATH = os.path.join(SAVED_MODELS_DIR, "event_hybrid_label_encoder.pkl")


class EventClassifierHybrid:
    """
    Concatenates TF-IDF + FinBERT [CLS] features, trains LR head.

    Features at training time
    --------------------------
    X_combined = hstack([X_tfidf,  X_finbert])
                shape = (n, ~10000 + 768)
    """

    CONFIDENCE_THRESHOLD = 0.35
    MIN_WORDS = 5

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
        embedder: FinBERTEmbedder | None = None,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            C=C,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.label_encoder = LabelEncoder()
        self.embedder = embedder if embedder is not None else FinBERTEmbedder()
        self.is_trained = False
        logger.info("EventClassifierHybrid initialised (C=%.3f, class_weight=%s).",
                    C, class_weight)

    # ----- Feature construction --------------------------------------------

    def _build_features(self, texts: list[str], fit_vectorizer: bool):
        """Build hstacked sparse TF-IDF + dense FinBERT embeddings."""
        if fit_vectorizer:
            X_tfidf = self.vectorizer.fit_transform(texts)
        else:
            X_tfidf = self.vectorizer.transform(texts)
        X_emb = self.embedder.extract_batch(texts)              # (n, 768) dense
        X_emb_sparse = csr_matrix(X_emb)                        # convert to sparse for hstack
        return hstack([X_tfidf, X_emb_sparse]).tocsr()

    # ----- Training ---------------------------------------------------------

    def train_event_classifier(self, X_train: list[str], y_train: list[str]) -> None:
        logger.info("Training EventClassifierHybrid on %d samples...", len(X_train))
        X_combined = self._build_features(X_train, fit_vectorizer=True)
        logger.info("  Combined feature shape: %s (TF-IDF + 768-dim FinBERT)",
                    X_combined.shape)

        y_int = self.label_encoder.fit_transform(y_train)
        logger.info("  Classes: %s", list(self.label_encoder.classes_))

        self.classifier.fit(X_combined, y_int)
        self.is_trained = True

        train_acc = self.classifier.score(X_combined, y_int)
        logger.info("  Training accuracy: %.4f", train_acc)

        # Persist embedding cache so retraining is fast
        self.embedder.save_cache()
        logger.info("Hybrid training complete.")

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
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")

        if strict and not self._is_meaningful_text(text, self.MIN_WORDS):
            return "unclassified"

        X = self._build_features([text], fit_vectorizer=False)

        if strict:
            # The TF-IDF half may be all zeros for OOV input — flag it.
            tfidf_only = self.vectorizer.transform([text])
            if tfidf_only.nnz == 0:
                return "unclassified"
            probas = self.classifier.predict_proba(X)[0]
            if float(np.max(probas)) < self.CONFIDENCE_THRESHOLD:
                return "unclassified"

        pred_int = int(self.classifier.predict(X)[0])
        return self.label_encoder.inverse_transform([pred_int])[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X = self._build_features([text], fit_vectorizer=False)
        probas = self.classifier.predict_proba(X)[0]
        classes = self.label_encoder.classes_
        return {str(cls): float(f"{float(prob):.4f}") for cls, prob in zip(classes, probas)}

    @property
    def classes_(self):
        if not self.is_trained:
            return np.array([])
        return np.asarray(self.label_encoder.classes_)

    # ----- Save / Load -------------------------------------------------------

    def save_model(
        self,
        clf_path: str = EVENT_HYBRID_PATH,
        vec_path: str = TFIDF_EVENT_HYBRID_PATH,
        label_path: str = EVENT_HYBRID_LABELENC_PATH,
    ) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not trained.")
        os.makedirs(os.path.dirname(clf_path), exist_ok=True)
        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.label_encoder, label_path)
        # Persist any newly-extracted embeddings
        self.embedder.save_cache()

        logger.info("EventClassifierHybrid saved:")
        logger.info("  Classifier      : %s", clf_path)
        logger.info("  TF-IDF vectorizer: %s", vec_path)
        logger.info("  Label encoder   : %s", label_path)

    def load_model(
        self,
        clf_path: str = EVENT_HYBRID_PATH,
        vec_path: str = TFIDF_EVENT_HYBRID_PATH,
        label_path: str = EVENT_HYBRID_LABELENC_PATH,
    ) -> None:
        for p in (clf_path, vec_path, label_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing artifact: {p}")
        self.classifier = joblib.load(clf_path)
        self.vectorizer = joblib.load(vec_path)
        self.label_encoder = joblib.load(label_path)
        # Embedder is constructed in __init__ — its cache loads automatically
        self.is_trained = True
        logger.info("EventClassifierHybrid loaded from %s", clf_path)

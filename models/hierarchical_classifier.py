"""
hierarchical_classifier.py — Two-Stage Coarse→Fine Event Classifier
=====================================================================

Stage 1: Classify into 4 coarse groups (~85-90% accuracy)
Stage 2: Per-group fine classifier (2-3 classes each)

Coarse Groups (updated for 7-class taxonomy):
    Financial_Performance  → Earnings
    Corporate_Action       → Mergers_Acquisitions, Legal_Action, Leadership_Change
    Market_External        → Market_Movement
    Business_Operations    → Product_Announcement, Regulatory_Action
"""

import os
import sys
import logging

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")

# ---------------------------------------------------------------------------
# Coarse → Fine mapping
# ---------------------------------------------------------------------------

COARSE_TO_FINE: dict[str, list[str]] = {
    "Financial_Performance": ["Earnings"],
    "Corporate_Action": ["Mergers_Acquisitions", "Legal_Action", "Leadership_Change"],
    "Market_External": ["Market_Movement"],
    "Business_Operations": ["Product_Announcement", "Regulatory_Action"],
}

FINE_TO_COARSE: dict[str, str] = {}
for coarse, fines in COARSE_TO_FINE.items():
    for fine in fines:
        FINE_TO_COARSE[fine] = coarse


class HierarchicalEventClassifier:
    """
    Two-stage hierarchical event classifier.

    Stage 1 classifies text into one of 4 coarse groups.
    Stage 2 uses a per-group classifier to pick the fine-grained label.
    For single-class groups (Financial_Performance), Stage 2 is skipped.
    """

    def __init__(self, tfidf_params: dict | None = None):
        default_tfidf = dict(
            max_features=10000, ngram_range=(1, 2), stop_words="english",
            min_df=2, max_df=0.95, sublinear_tf=True,
        )
        self.tfidf_params = {**default_tfidf, **(tfidf_params or {})}
        self.lr_params = dict(
            max_iter=2000, solver="lbfgs", C=1.0,
            class_weight="balanced", random_state=42,
        )

        # Stage 1: coarse classifier
        self.coarse_vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.coarse_classifier = LogisticRegression(**self.lr_params)

        # Stage 2: per-group fine classifiers (only for multi-class groups)
        self.fine_vectorizers: dict[str, TfidfVectorizer] = {}
        self.fine_classifiers: dict[str, LogisticRegression] = {}

        self.is_trained = False

    def train(self, X_train: list[str], y_train: list[str]) -> None:
        """Train both stages of the hierarchical classifier."""
        logger.info("Training hierarchical event classifier on %d samples...", len(X_train))

        # Map fine labels → coarse
        y_coarse = [FINE_TO_COARSE.get(y, "Business_Operations") for y in y_train]

        # Stage 1: train coarse classifier
        X_tfidf = self.coarse_vectorizer.fit_transform(X_train)
        self.coarse_classifier.fit(X_tfidf, y_coarse)
        coarse_acc = self.coarse_classifier.score(X_tfidf, y_coarse)
        logger.info("  Coarse classifier training acc: %.4f", coarse_acc)

        # Stage 2: train fine classifiers for multi-class groups
        for coarse_group, fine_classes in COARSE_TO_FINE.items():
            if len(fine_classes) <= 1:
                # Single-class group — no fine classifier needed
                continue

            # Filter to samples in this coarse group
            mask = [y_coarse[i] == coarse_group for i in range(len(y_coarse))]
            X_group = [X_train[i] for i in range(len(X_train)) if mask[i]]
            y_group = [y_train[i] for i in range(len(y_train)) if mask[i]]

            if len(set(y_group)) < 2:
                logger.warning("  Group '%s' has <2 classes, skipping fine classifier", coarse_group)
                continue

            vec = TfidfVectorizer(**self.tfidf_params)
            clf = LogisticRegression(**self.lr_params)

            X_g_tfidf = vec.fit_transform(X_group)
            clf.fit(X_g_tfidf, y_group)

            self.fine_vectorizers[coarse_group] = vec
            self.fine_classifiers[coarse_group] = clf

            fine_acc = clf.score(X_g_tfidf, y_group)
            logger.info("  Fine classifier '%s' (%d samples, %d classes): train acc %.4f",
                         coarse_group, len(X_group), len(set(y_group)), fine_acc)

        self.is_trained = True
        logger.info("Hierarchical classifier training complete.")

    def predict(self, text: str) -> str:
        """Predict fine-grained event type via coarse→fine pipeline."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        # Stage 1: coarse prediction
        X_tfidf = self.coarse_vectorizer.transform([text])
        coarse_pred = self.coarse_classifier.predict(X_tfidf)[0]

        # Stage 2: fine prediction
        fine_classes = COARSE_TO_FINE.get(coarse_pred, ["Other"])
        if len(fine_classes) == 1:
            return fine_classes[0]

        if coarse_pred in self.fine_classifiers:
            vec = self.fine_vectorizers[coarse_pred]
            clf = self.fine_classifiers[coarse_pred]
            X_fine = vec.transform([text])
            return clf.predict(X_fine)[0]

        return fine_classes[0]

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]

    def save_model(self) -> None:
        """Save all components to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save — not trained.")

        base = os.path.join(SAVED_MODELS_DIR, "hierarchical")
        os.makedirs(base, exist_ok=True)

        joblib.dump(self.coarse_vectorizer, os.path.join(base, "coarse_vec.pkl"))
        joblib.dump(self.coarse_classifier, os.path.join(base, "coarse_clf.pkl"))

        for group in self.fine_classifiers:
            joblib.dump(self.fine_vectorizers[group],
                        os.path.join(base, f"fine_vec_{group}.pkl"))
            joblib.dump(self.fine_classifiers[group],
                        os.path.join(base, f"fine_clf_{group}.pkl"))

        logger.info("Hierarchical classifier saved to: %s", base)

    def load_model(self) -> None:
        """Load all components from disk."""
        base = os.path.join(SAVED_MODELS_DIR, "hierarchical")

        self.coarse_vectorizer = joblib.load(os.path.join(base, "coarse_vec.pkl"))
        self.coarse_classifier = joblib.load(os.path.join(base, "coarse_clf.pkl"))

        self.fine_vectorizers = {}
        self.fine_classifiers = {}
        for group in COARSE_TO_FINE:
            vec_path = os.path.join(base, f"fine_vec_{group}.pkl")
            clf_path = os.path.join(base, f"fine_clf_{group}.pkl")
            if os.path.isfile(vec_path) and os.path.isfile(clf_path):
                self.fine_vectorizers[group] = joblib.load(vec_path)
                self.fine_classifiers[group] = joblib.load(clf_path)

        self.is_trained = True
        logger.info("Hierarchical classifier loaded from: %s", base)

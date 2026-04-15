"""
relevance_classifier.py — TF-IDF + Logistic Regression Relevance Classifier
============================================================================

Binary classifier: predicts whether an article is **financially relevant**
(label = 1) or **not** (label = 0).

Trained on the full 5,466-row gold dataset produced by the seed Gemini pipeline
(both rel=0 and rel=1). The deterministic keyword filter in
`pipeline/run.py._keyword_filter()` still runs first as a hard pre-screen
(junk_stub / body_too_short / no_keyword_match), and only articles that pass
it are fed to this classifier.

API mirrors `models/event_classifier.py` so the trainer / pipeline can use
both interchangeably.

Saved artefacts:
    models/saved_models/relevance_classifier.pkl
    models/saved_models/tfidf_relevance_vectorizer.pkl
"""

import os
import sys
import logging

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
RELEVANCE_CLF_PATH = os.path.join(SAVED_MODELS_DIR, "relevance_classifier.pkl")
TFIDF_RELEVANCE_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_relevance_vectorizer.pkl")


class RelevanceClassifierLR:
    """Binary relevance classifier (1 = financially relevant, 0 = noise)."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
        self.is_trained = False
        logger.info("RelevanceClassifierLR initialised.")

    # ----- Training ---------------------------------------------------------

    def train(self, X_train: list[str], y_train: list[int]) -> None:
        logger.info("Training relevance classifier on %d samples...", len(X_train))
        X_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info("  TF-IDF matrix shape: %s", X_tfidf.shape)
        self.classifier.fit(X_tfidf, y_train)
        self.is_trained = True
        train_acc = self.classifier.score(X_tfidf, y_train)
        logger.info("  Training accuracy: %.4f", train_acc)

    # ----- Prediction -------------------------------------------------------

    def predict(self, text: str) -> int:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        X_tfidf = self.vectorizer.transform([text])
        return int(self.classifier.predict(X_tfidf)[0])

    def predict_proba(self, text: str) -> float:
        """Return probability that the article is relevant (class 1)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X_tfidf = self.vectorizer.transform([text])
        probas = self.classifier.predict_proba(X_tfidf)[0]
        # classes_ is sorted; index for class 1
        idx_pos = list(self.classifier.classes_).index(1)
        return float(probas[idx_pos])

    def predict_batch(self, texts: list[str]) -> list[int]:
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X_tfidf = self.vectorizer.transform(texts)
        return [int(p) for p in self.classifier.predict(X_tfidf)]

    def predict_proba_batch(self, texts: list[str]) -> list[float]:
        if not self.is_trained:
            raise RuntimeError("Model not trained or loaded.")
        X_tfidf = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(X_tfidf)
        idx_pos = list(self.classifier.classes_).index(1)
        return [float(p[idx_pos]) for p in probas]

    # ----- Save / Load -------------------------------------------------------

    def save_model(
        self,
        clf_path: str = RELEVANCE_CLF_PATH,
        vec_path: str = TFIDF_RELEVANCE_PATH,
    ) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not trained.")
        os.makedirs(os.path.dirname(clf_path), exist_ok=True)
        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.vectorizer, vec_path)
        logger.info("Relevance classifier saved to: %s", clf_path)
        logger.info("TF-IDF vectorizer saved to: %s", vec_path)

    def load_model(
        self,
        clf_path: str = RELEVANCE_CLF_PATH,
        vec_path: str = TFIDF_RELEVANCE_PATH,
    ) -> None:
        if not os.path.isfile(clf_path):
            raise FileNotFoundError(f"Relevance classifier not found: {clf_path}")
        if not os.path.isfile(vec_path):
            raise FileNotFoundError(f"Vectorizer not found: {vec_path}")
        self.classifier = joblib.load(clf_path)
        self.vectorizer = joblib.load(vec_path)
        self.is_trained = True
        logger.info("Relevance classifier loaded from: %s", clf_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    sample_texts = [
        "apple reports record quarterly earnings beating analyst expectations",
        "rcb defeats csk by 6 wickets in ipl 2026 final at chinnaswamy",
        "fed cuts interest rates by 25 basis points amid easing inflation",
        "bollywood actor announces new film release next summer",
        "reliance industries posts 12 percent rise in q3 net profit",
        "world cup football match draws record viewers globally",
    ]
    sample_labels = [1, 0, 1, 0, 1, 0]

    clf = RelevanceClassifierLR()
    clf.vectorizer.set_params(min_df=1)
    clf.train(sample_texts, sample_labels)

    test = "tata motors q4 revenue jumps 18 percent on strong domestic sales"
    print(f"\nInput : {test}")
    print(f"Pred  : {clf.predict(test)}  (proba_relevant = {clf.predict_proba(test):.4f})")

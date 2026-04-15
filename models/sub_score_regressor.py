"""
sub_score_regressor.py — V3 Four-Sub-Score Impact Model
========================================================

Trains four Ridge regressors on the 1,817 gold-labelled articles, one per
sub-score:

    materiality
    market_linkage
    time_sensitivity
    credibility

Each regressor predicts its sub-score in [0.0, 1.0] from the article text.
The final ``impact_score`` is then computed deterministically by the
formula in ``docs/pipeline-algorithm-spec.md`` § 3:

    impact_score = (
        0.4 * materiality
      + 0.3 * market_linkage
      + 0.2 * time_sensitivity
      + 0.1 * credibility
    )

This mirrors V2's hybrid LLM-grades-subscores + Python-computes-formula
contract. We keep the LLM out of runtime entirely; the four trained
regressors take its place.

A single shared TF-IDF vectorizer is fitted once and reused across the
four regressors — cheaper to train, lighter to ship, and ensures all
four sub-scores see the same feature space (which keeps the
escalation safety check on inconsistent sub-scores meaningful).
"""

from __future__ import annotations

import logging
import os
import sys

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")

SUB_SCORES = ("materiality", "market_linkage", "time_sensitivity", "credibility")
WEIGHTS = {
    "materiality":      0.4,
    "market_linkage":   0.3,
    "time_sensitivity": 0.2,
    "credibility":      0.1,
}

TFIDF_SUBSCORE_PATH = os.path.join(SAVED_MODELS_DIR, "tfidf_subscore_vectorizer.pkl")
SUBSCORE_PATHS = {
    name: os.path.join(SAVED_MODELS_DIR, f"subscore_{name}.pkl")
    for name in SUB_SCORES
}


def compute_impact_score(sub_scores: dict[str, float]) -> float:
    """Apply the canonical weighted-sum formula. Returns clamped [0,1]."""
    raw = sum(WEIGHTS[k] * float(sub_scores.get(k, 0.0)) for k in SUB_SCORES)
    return float(round(max(0.0, min(1.0, raw)), 4))


class SubScoreRegressor:
    """Four Ridge regressors + one shared TF-IDF vectorizer."""

    def __init__(self, alpha: float = 1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.regressors: dict[str, Ridge] = {
            name: Ridge(alpha=alpha, random_state=42) for name in SUB_SCORES
        }
        self.is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: list[str],
        sub_score_targets: dict[str, list[float]],
    ) -> None:
        """
        Fit the shared TF-IDF on X_train, then train one Ridge per sub-score.

        Parameters
        ----------
        X_train : list[str]
            Cleaned article texts.
        sub_score_targets : dict[str, list[float]]
            Mapping of sub-score name → list of float targets in [0, 1].
            Must contain all four keys in SUB_SCORES.
        """
        missing = [k for k in SUB_SCORES if k not in sub_score_targets]
        if missing:
            raise ValueError(f"Missing sub-score targets: {missing}")

        logger.info("Training SubScoreRegressor on %d samples", len(X_train))
        X_tfidf = self.vectorizer.fit_transform(X_train)
        logger.info("  TF-IDF matrix shape: %s", X_tfidf.shape)

        for name in SUB_SCORES:
            y = np.asarray(sub_score_targets[name], dtype=float)
            self.regressors[name].fit(X_tfidf, y)
            r2 = self.regressors[name].score(X_tfidf, y)
            logger.info("  %-18s training R² = %.4f", name, r2)

        self.is_trained = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _clamp(self, v: float) -> float:
        return float(max(0.0, min(1.0, float(v))))

    def predict_subscores(self, text: str) -> dict[str, float]:
        """Return {sub_score_name: predicted_value} for one article."""
        if not self.is_trained:
            raise RuntimeError("Not trained. Call train() or load_model().")
        X = self.vectorizer.transform([text])
        return {
            name: self._clamp(self.regressors[name].predict(X)[0])
            for name in SUB_SCORES
        }

    def predict(self, text: str) -> dict[str, float]:
        """
        Predict the four sub-scores AND the deterministic impact_score.

        Returns
        -------
        dict
            {
              "materiality": float,
              "market_linkage": float,
              "time_sensitivity": float,
              "credibility": float,
              "impact_score": float,   # weighted-sum, formula in module docstring
            }
        """
        subs = self.predict_subscores(text)
        subs["impact_score"] = compute_impact_score(subs)
        return subs

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        if not self.is_trained:
            raise RuntimeError("Not trained.")
        X = self.vectorizer.transform(texts)
        # Predict each sub-score in one shot for efficiency
        all_preds = {
            name: np.clip(self.regressors[name].predict(X), 0.0, 1.0)
            for name in SUB_SCORES
        }
        out: list[dict[str, float]] = []
        for i in range(len(texts)):
            row = {name: float(round(all_preds[name][i], 4)) for name in SUB_SCORES}
            row["impact_score"] = compute_impact_score(row)
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Cannot save — not trained.")
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        joblib.dump(self.vectorizer, TFIDF_SUBSCORE_PATH)
        for name, path in SUBSCORE_PATHS.items():
            joblib.dump(self.regressors[name], path)
        logger.info("SubScoreRegressor saved (vectorizer + 4 regressors)")

    def load_model(self) -> None:
        if not os.path.isfile(TFIDF_SUBSCORE_PATH):
            raise FileNotFoundError(f"Vectorizer not found: {TFIDF_SUBSCORE_PATH}")
        self.vectorizer = joblib.load(TFIDF_SUBSCORE_PATH)
        for name, path in SUBSCORE_PATHS.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Regressor not found: {path}")
            self.regressors[name] = joblib.load(path)
        self.is_trained = True
        logger.info("SubScoreRegressor loaded from disk")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    sample_texts = [
        "reliance industries reports record q4 net profit dividend announced 11 rupees per share",
        "sensex slips 200 points on profit booking ahead of fed meeting",
        "tata motors unveils ev hatchback at auto expo 2026 with launch in q3",
        "rbi imposes monetary penalty on hdfc bank for kyc lapses in retail branches",
        "anonymous twitter rumour suggests possible takeover of midcap pharma firm",
    ]
    sample_targets = {
        "materiality":      [0.95, 0.55, 0.65, 0.50, 0.20],
        "market_linkage":   [1.00, 0.90, 0.85, 0.85, 0.40],
        "time_sensitivity": [0.95, 0.90, 0.55, 0.85, 0.50],
        "credibility":      [0.95, 0.90, 0.85, 0.95, 0.20],
    }

    reg = SubScoreRegressor()
    reg.vectorizer.set_params(min_df=1, max_df=1.0)
    reg.train(sample_texts, sample_targets)

    print("\nSub-score prediction smoke test")
    print("-" * 60)
    test = "infosys posts strong q4 net profit and announces large buyback"
    out = reg.predict(test)
    for k, v in out.items():
        print(f"  {k:18s} {v:.4f}")

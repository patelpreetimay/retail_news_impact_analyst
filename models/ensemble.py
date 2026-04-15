"""
ensemble.py — Weighted Ensemble Classifier for RNIA
====================================================

Combines TF-IDF+LR probabilities with keyword-rule scores via weighted voting.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
ENSEMBLE_CONFIG_PATH = os.path.join(SAVED_MODELS_DIR, "ensemble_config.json")


class EnsembleEventClassifier:
    """Ensemble event classifier: TF-IDF+LR + keyword rules with weighted fusion."""

    def __init__(self, lr_model, keyword_model, kw_weight: float = 0.25):
        self.lr_model = lr_model
        self.keyword_model = keyword_model
        self.kw_weight = kw_weight
        self.lr_weight = 1.0 - kw_weight

    def predict(self, text: str) -> str:
        lr_probs = self.lr_model.predict_proba(text)
        kw_scores = self.keyword_model.predict_scores(text)
        kw_total = sum(kw_scores.values())
        if kw_total == 0:
            return max(lr_probs, key=lr_probs.get)
        all_classes = sorted(set(lr_probs.keys()) | set(kw_scores.keys()))
        combined = {}
        for cls in all_classes:
            combined[cls] = self.lr_weight * lr_probs.get(cls, 0.0) + self.kw_weight * kw_scores.get(cls, 0.0)
        return max(combined, key=combined.get)

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]

    def calibrate(self, X_val: list[str], y_val: list[str],
                  weights_to_try: Optional[list[float]] = None) -> float:
        if weights_to_try is None:
            weights_to_try = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        best_acc, best_w = -1.0, 0.0
        for w in weights_to_try:
            self.kw_weight = w
            self.lr_weight = 1.0 - w
            preds = self.predict_batch(X_val)
            acc = sum(1 for p, t in zip(preds, y_val) if p == t) / len(y_val)
            if acc > best_acc:
                best_acc, best_w = acc, w
        self.kw_weight = best_w
        self.lr_weight = 1.0 - best_w
        logger.info("Ensemble event calibration: kw_weight=%.2f val_acc=%.4f", best_w, best_acc)
        return best_w


class EnsembleStanceClassifier:
    """Ensemble stance classifier: TF-IDF+LR + keyword rules with weighted fusion."""

    def __init__(self, lr_model, keyword_model, kw_weight: float = 0.25):
        self.lr_model = lr_model
        self.keyword_model = keyword_model
        self.kw_weight = kw_weight
        self.lr_weight = 1.0 - kw_weight

    def predict(self, text: str) -> str:
        lr_probs = self.lr_model.predict_proba(text)
        kw_scores = self.keyword_model.predict_scores(text)
        kw_total = sum(kw_scores.values())
        if kw_total == 0:
            return max(lr_probs, key=lr_probs.get)
        all_classes = sorted(set(lr_probs.keys()) | set(kw_scores.keys()))
        combined = {}
        for cls in all_classes:
            combined[cls] = self.lr_weight * lr_probs.get(cls, 0.0) + self.kw_weight * kw_scores.get(cls, 0.0)
        return max(combined, key=combined.get)

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]

    def calibrate(self, X_val: list[str], y_val: list[str],
                  weights_to_try: Optional[list[float]] = None) -> float:
        if weights_to_try is None:
            weights_to_try = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        best_acc, best_w = -1.0, 0.0
        for w in weights_to_try:
            self.kw_weight = w
            self.lr_weight = 1.0 - w
            preds = self.predict_batch(X_val)
            acc = sum(1 for p, t in zip(preds, y_val) if p == t) / len(y_val)
            if acc > best_acc:
                best_acc, best_w = acc, w
        self.kw_weight = best_w
        self.lr_weight = 1.0 - best_w
        logger.info("Ensemble stance calibration: kw_weight=%.2f val_acc=%.4f", best_w, best_acc)
        return best_w


def save_ensemble_config(event_kw_weight: float, stance_kw_weight: float,
                          path: str = ENSEMBLE_CONFIG_PATH) -> None:
    config = {"event_kw_weight": event_kw_weight, "stance_kw_weight": stance_kw_weight}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Ensemble config saved to: %s", path)


def load_ensemble_config(path: str = ENSEMBLE_CONFIG_PATH) -> dict:
    if not os.path.isfile(path):
        return {"event_kw_weight": 0.25, "stance_kw_weight": 0.25}
    with open(path) as f:
        return json.load(f)

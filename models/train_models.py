"""
train_models.py — Enhanced Model Training Pipeline for RNIA
============================================================

Improvements over baseline:
  - Stance collapse: 4→3 classes (mixed→neutral)
  - SMOTE oversampling for minority event classes
  - Expanded dataset support (gold + silver labels)
  - Optuna best-hyperparams loading
  - 5-fold stratified CV evaluation
  - Keyword-rule ensemble with calibrated weights
  - Hierarchical event classifier comparison
"""

import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    mean_absolute_error, r2_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.event_classifier import EventClassifierLR
from models.stance_detector import StanceDetectorLR
from models.impact_regressor import ImpactRegressor
from models.relevance_classifier import RelevanceClassifierLR
from models.sub_score_regressor import SUB_SCORES, SubScoreRegressor
from models.keyword_rules import KeywordEventClassifier, KeywordStanceClassifier
from models.ensemble import (
    EnsembleEventClassifier, EnsembleStanceClassifier,
    save_ensemble_config,
)
from models.hierarchical_classifier import HierarchicalEventClassifier
from utils.text_features import (
    build_input_texts,
    save_text_features_config,
    get_headline_weight,
    DEFAULT_HEADLINE_WEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LABELED_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv"
)
EXPANDED_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled_expanded.csv"
)
RELEVANCE_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "relevance_dataset.csv"
)
HYPERPARAMS_PATH = os.path.join(
    PROJECT_ROOT, "models", "saved_models", "best_hyperparams.json"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42
STANCE_COLLAPSE = {"mixed": "neutral"}

# Import the canonical 9→7 event collapse from the taxonomy
from taxonomy.event_taxonomy import EVENT_COLLAPSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_labeled_data(use_expanded: bool = True) -> pd.DataFrame:
    """Load labeled dataset, preferring expanded if available."""
    path = EXPANDED_DATASET if (use_expanded and os.path.isfile(EXPANDED_DATASET)) else LABELED_DATASET
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Labeled dataset not found at:\n  {path}\n"
            "Please run scripts/export_gold_dataset.py first."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d labeled articles from: %s", len(df), path)
    return df


def load_best_hyperparams() -> dict:
    """Load Optuna best hyperparams if available."""
    if os.path.isfile(HYPERPARAMS_PATH):
        with open(HYPERPARAMS_PATH) as f:
            params = json.load(f)
        logger.info("Loaded best hyperparams from: %s", HYPERPARAMS_PATH)
        return params
    return {}


def apply_smote(X_tfidf, y_labels):
    """Apply SMOTE to balance minority classes in TF-IDF feature space."""
    try:
        from imblearn.over_sampling import SMOTE
        from collections import Counter
    except ImportError:
        logger.warning("imbalanced-learn not installed, skipping SMOTE")
        return X_tfidf, y_labels

    counts = Counter(y_labels)
    min_count = min(counts.values())
    if min_count < 2:
        logger.warning("Some classes have <2 samples, skipping SMOTE")
        return X_tfidf, y_labels

    k = min(5, min_count - 1)
    smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_tfidf, y_labels)

    old_counts = dict(sorted(counts.items(), key=lambda x: x[1]))
    new_counts = dict(sorted(Counter(y_res).items(), key=lambda x: x[1]))
    logger.info("SMOTE: %d→%d samples (k=%d)", X_tfidf.shape[0], X_res.shape[0], k)
    for cls in old_counts:
        if old_counts[cls] != new_counts.get(cls, 0):
            logger.info("  %s: %d → %d", cls, old_counts[cls], new_counts[cls])

    return X_res, y_res


def apply_hyperparams(model, params: dict, model_type: str):
    """Apply Optuna best hyperparams to a model's vectorizer and classifier."""
    hp = params.get(model_type, {}).get("best_params", {})
    if not hp:
        return

    vec_params = {}
    clf_params = {}
    for k, v in hp.items():
        if k == "ngram_max":
            vec_params["ngram_range"] = (1, v)
        elif k in ("max_features", "min_df", "max_df", "sublinear_tf"):
            vec_params[k] = v
        elif k == "C":
            clf_params[k] = v

    if vec_params:
        model.vectorizer.set_params(**vec_params)
        logger.info("  Applied TF-IDF params: %s", vec_params)
    if clf_params:
        model.classifier.set_params(**clf_params)
        logger.info("  Applied LR params: %s", clf_params)


def evaluate_5fold_cv(X, y, model_class, model_type: str, hyperparams: dict,
                       use_smote: bool = False, label: str = ""):
    """Run 5-fold stratified CV and return mean metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        model = model_class()
        apply_hyperparams(model, hyperparams, model_type)

        if len(X_train) < 100:
            model.vectorizer.set_params(min_df=1, max_df=1.0)

        # Train
        X_tfidf = model.vectorizer.fit_transform(X_train)
        y_tr = y_train

        if use_smote:
            X_tfidf, y_tr = apply_smote(X_tfidf, y_tr)

        model.classifier.fit(X_tfidf, y_tr)
        model.is_trained = True

        # Evaluate
        X_test_tfidf = model.vectorizer.transform(X_test)
        y_pred = model.classifier.predict(X_test_tfidf).tolist()

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    std_acc = np.std(accs)
    logger.info("  %s 5-fold CV: acc=%.4f±%.4f  macro-F1=%.4f",
                label, mean_acc, std_acc, mean_f1)
    return mean_acc, mean_f1


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def train_and_evaluate(
    use_expanded: bool = True,
    use_smote: bool = True,
    headline_weight: int | None = None,
    extra_event_model: str | None = None,
):
    """
    Full enhanced training pipeline.

    Parameters
    ----------
    use_expanded : bool
        Load gold+silver dataset when available.
    use_smote : bool
        Apply SMOTE to the event classifier's training set.
    headline_weight : int | None
        Number of times the headline is repeated when constructing input
        text. ``1`` = legacy behavior (no weighting). ``3`` = recommended.
        Persisted to ``text_features_config.json`` for inference parity.

        **If None (the default), reads the last-saved value from disk.**
        That way, once you've trained with weight=3 once via CLI, every
        subsequent menu-driven [3] / [7] reuses weight=3 automatically.
        Fresh installs (no config file yet) fall back to 1 = legacy.
    extra_event_model : {None, "xgb", "hybrid", "both"}
        Optionally train an additional event classifier alongside the LR
        baseline. The LR baseline is ALWAYS trained — extras are saved
        under separate filenames for opt-in use in the runtime pipeline.
    """
    # Resolve headline_weight: explicit arg wins, else fall back to config,
    # else legacy default (1). This makes menu [3]/[7] preserve the last
    # CLI-trained setting without the user needing to remember.
    if headline_weight is None:
        headline_weight = get_headline_weight(default=DEFAULT_HEADLINE_WEIGHT)

    logger.info("=" * 60)
    logger.info("RNIA — Enhanced Model Training Pipeline")
    logger.info("  headline_weight=%d  extra_event_model=%s",
                headline_weight, extra_event_model or "none")
    logger.info("=" * 60)

    # ---- Load data ----------------------------------------------------------
    df = load_labeled_data(use_expanded)
    df = df.dropna(subset=["clean_text", "event_type", "stance"]).reset_index(drop=True)

    # Step 2: Collapse stance 4→3
    df["stance"] = df["stance"].replace(STANCE_COLLAPSE)
    logger.info("Stance collapsed to 3 classes: %s",
                dict(df["stance"].value_counts()))

    # Step 2b: Collapse event types 9→7
    df["event_type"] = df["event_type"].replace(EVENT_COLLAPSE)
    logger.info("Event types collapsed to 7 classes: %s",
                dict(df["event_type"].value_counts()))

    # Remove Unclassified event (only 1 sample)
    df = df[df["event_type"] != "Unclassified"].reset_index(drop=True)
    logger.info("Articles after cleanup: %d", len(df))

    if len(df) < 10:
        logger.error("Need at least 10 labeled articles. Found %d.", len(df))
        return

    # ---- Build model input text (with optional headline weighting) ----------
    if "headline" not in df.columns:
        df["headline"] = ""
    df["headline"] = df["headline"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")
    X = build_input_texts(
        df["headline"].tolist(),
        df["clean_text"].tolist(),
        weight=headline_weight,
    )
    logger.info("Built model input texts (headline_weight=%d)", headline_weight)

    # Persist headline_weight so inference paths use the same construction
    save_text_features_config(headline_weight)

    y_event = df["event_type"].tolist()
    y_stance = df["stance"].tolist()
    y_impact = df["impact_score"].tolist() if "impact_score" in df.columns else None

    # Load best hyperparams
    hyperparams = load_best_hyperparams()

    # ---- 5-Fold CV Evaluation -----------------------------------------------
    print("\n" + "=" * 60)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    ev_acc, ev_f1 = evaluate_5fold_cv(
        X, y_event, EventClassifierLR, "event", hyperparams,
        use_smote=use_smote, label="Event Classifier"
    )
    st_acc, st_f1 = evaluate_5fold_cv(
        X, y_stance, StanceDetectorLR, "stance", hyperparams,
        use_smote=False, label="Stance Detector"
    )

    # ---- Train/Test Split for final models ----------------------------------
    try:
        (X_train, X_test, y_event_train, y_event_test,
         y_stance_train, y_stance_test) = train_test_split(
            X, y_event, y_stance,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_event,
        )
    except ValueError:
        logger.warning("Stratified split failed, using non-stratified.")
        (X_train, X_test, y_event_train, y_event_test,
         y_stance_train, y_stance_test) = train_test_split(
            X, y_event, y_stance,
            test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )

    if y_impact is not None:
        _, _, y_impact_train, y_impact_test = train_test_split(
            X, y_impact, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
    else:
        y_impact_train = y_impact_test = None

    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # ---- Train Event Classifier (with SMOTE) --------------------------------
    print("\n" + "-" * 60)
    print("Training EVENT CLASSIFIER (7 categories + SMOTE)")
    print("-" * 60)

    event_clf = EventClassifierLR()
    apply_hyperparams(event_clf, hyperparams, "event")
    if len(X_train) < 100:
        event_clf.vectorizer.set_params(min_df=1, max_df=1.0)

    X_event_tfidf = event_clf.vectorizer.fit_transform(X_train)

    if use_smote:
        X_event_smote, y_event_smote = apply_smote(X_event_tfidf, y_event_train)
    else:
        X_event_smote, y_event_smote = X_event_tfidf, y_event_train

    event_clf.classifier.fit(X_event_smote, y_event_smote)
    event_clf.is_trained = True

    y_event_pred = event_clf.classifier.predict(
        event_clf.vectorizer.transform(X_test)
    ).tolist()
    event_accuracy = accuracy_score(y_event_test, y_event_pred)
    event_f1 = f1_score(y_event_test, y_event_pred, average="macro", zero_division=0)

    print(f"\nAccuracy: {event_accuracy:.4f}  Macro-F1: {event_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_event_test, y_event_pred, zero_division=0))

    # ---- Train Stance Detector ----------------------------------------------
    print("-" * 60)
    print("Training STANCE DETECTOR (3 classes)")
    print("-" * 60)

    stance_det = StanceDetectorLR()
    apply_hyperparams(stance_det, hyperparams, "stance")
    if len(X_train) < 100:
        stance_det.vectorizer.set_params(min_df=1, max_df=1.0)

    stance_det.train_stance_model(X_train, y_stance_train)

    y_stance_pred = [stance_det.predict_stance(t) for t in X_test]
    stance_accuracy = accuracy_score(y_stance_test, y_stance_pred)
    stance_f1 = f1_score(y_stance_test, y_stance_pred, average="macro", zero_division=0)

    print(f"\nAccuracy: {stance_accuracy:.4f}  Macro-F1: {stance_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_stance_test, y_stance_pred, zero_division=0))

    # ---- Train Hierarchical Classifier (Step 7) -----------------------------
    print("-" * 60)
    print("Training HIERARCHICAL EVENT CLASSIFIER")
    print("-" * 60)

    hier_clf = HierarchicalEventClassifier()
    hier_clf.train(X_train, y_event_train)

    y_hier_pred = hier_clf.predict_batch(X_test)
    hier_accuracy = accuracy_score(y_event_test, y_hier_pred)
    hier_f1 = f1_score(y_event_test, y_hier_pred, average="macro", zero_division=0)

    print(f"\nHierarchical Accuracy: {hier_accuracy:.4f}  Macro-F1: {hier_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_event_test, y_hier_pred, zero_division=0))

    # ---- Ensemble (Step 6) --------------------------------------------------
    print("-" * 60)
    print("Calibrating ENSEMBLE CLASSIFIERS")
    print("-" * 60)

    kw_event = KeywordEventClassifier()
    kw_stance = KeywordStanceClassifier()

    ens_event = EnsembleEventClassifier(event_clf, kw_event)
    ens_event_w = ens_event.calibrate(X_test, y_event_test)

    ens_stance = EnsembleStanceClassifier(stance_det, kw_stance)
    ens_stance_w = ens_stance.calibrate(X_test, y_stance_test)

    y_ens_event = ens_event.predict_batch(X_test)
    ens_event_acc = accuracy_score(y_event_test, y_ens_event)
    ens_event_f1 = f1_score(y_event_test, y_ens_event, average="macro", zero_division=0)

    y_ens_stance = ens_stance.predict_batch(X_test)
    ens_stance_acc = accuracy_score(y_stance_test, y_ens_stance)
    ens_stance_f1 = f1_score(y_stance_test, y_ens_stance, average="macro", zero_division=0)

    print(f"\nEnsemble Event:  acc={ens_event_acc:.4f}  F1={ens_event_f1:.4f}  kw_w={ens_event_w:.2f}")
    print(f"Ensemble Stance: acc={ens_stance_acc:.4f}  F1={ens_stance_f1:.4f}  kw_w={ens_stance_w:.2f}")

    save_ensemble_config(ens_event_w, ens_stance_w)

    # ---- Optional: extra event classifiers (XGBoost / Hybrid) ---------------
    extra_results: dict[str, tuple[float, float]] = {}

    train_xgb = extra_event_model in ("xgb", "both")
    train_hybrid = extra_event_model in ("hybrid", "both")

    if train_xgb:
        print("\n" + "-" * 60)
        print("Training EXTRA: EventClassifierXGB (TF-IDF + XGBoost)")
        print("-" * 60)
        try:
            from models.event_classifier_xgb import EventClassifierXGB
            xgb_clf = EventClassifierXGB()
            if len(X_train) < 100:
                xgb_clf.vectorizer.set_params(min_df=1, max_df=1.0)
            X_xgb_tfidf = xgb_clf.vectorizer.fit_transform(X_train)
            y_xgb_int = xgb_clf.label_encoder.fit_transform(y_event_train)

            if use_smote:
                X_xgb_smote, y_xgb_smote = apply_smote(X_xgb_tfidf, y_xgb_int)
            else:
                X_xgb_smote, y_xgb_smote = X_xgb_tfidf, y_xgb_int

            xgb_clf.classifier.fit(X_xgb_smote, y_xgb_smote)
            xgb_clf.is_trained = True

            y_xgb_pred_int = xgb_clf.classifier.predict(
                xgb_clf.vectorizer.transform(X_test)
            )
            y_xgb_pred = xgb_clf.label_encoder.inverse_transform(y_xgb_pred_int).tolist()
            xgb_acc = accuracy_score(y_event_test, y_xgb_pred)
            xgb_f1 = f1_score(y_event_test, y_xgb_pred, average="macro", zero_division=0)
            extra_results["XGBoost"] = (xgb_acc, xgb_f1)
            print(f"\nAccuracy: {xgb_acc:.4f}  Macro-F1: {xgb_f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_event_test, y_xgb_pred, zero_division=0))
            xgb_clf.save_model()
        except ImportError as e:
            logger.error("XGBoost training skipped — package missing: %s", e)
            logger.error("Install with:  pip install xgboost")

    if train_hybrid:
        print("\n" + "-" * 60)
        print("Training EXTRA: EventClassifierHybrid (TF-IDF + FinBERT embeddings + LR)")
        print("-" * 60)
        try:
            from models.event_classifier_hybrid import EventClassifierHybrid
            hyb_clf = EventClassifierHybrid()
            if len(X_train) < 100:
                hyb_clf.vectorizer.set_params(min_df=1, max_df=1.0)
            hyb_clf.train_event_classifier(X_train, y_event_train)

            y_hyb_pred = [hyb_clf.predict_event(t) for t in X_test]
            hyb_acc = accuracy_score(y_event_test, y_hyb_pred)
            hyb_f1 = f1_score(y_event_test, y_hyb_pred, average="macro", zero_division=0)
            extra_results["Hybrid (TFIDF+FinBERT)"] = (hyb_acc, hyb_f1)
            print(f"\nAccuracy: {hyb_acc:.4f}  Macro-F1: {hyb_f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_event_test, y_hyb_pred, zero_division=0))
            hyb_clf.save_model()
        except ImportError as e:
            logger.error("Hybrid training skipped — package missing: %s", e)
            logger.error("Install with:  pip install transformers torch")

    # ---- Impact Regressor ---------------------------------------------------
    impact_reg = None
    if y_impact_train is not None:
        print("\n" + "-" * 60)
        print("Training IMPACT REGRESSOR")
        print("-" * 60)

        impact_reg = ImpactRegressor()
        if len(X_train) < 100:
            impact_reg.vectorizer.set_params(min_df=1, max_df=1.0)
        impact_reg.train(X_train, y_impact_train)

        y_impact_pred = [impact_reg.predict(t) for t in X_test]
        mae = mean_absolute_error(y_impact_test, y_impact_pred)
        r2 = r2_score(y_impact_test, y_impact_pred)
        print(f"\nMAE: {mae:.4f}  R²: {r2:.4f}")

    # ---- Save Models --------------------------------------------------------
    print("\n" + "-" * 60)
    print("Saving trained models...")
    print("-" * 60)

    event_clf.save_model()
    stance_det.save_model()
    hier_clf.save_model()
    if impact_reg is not None:
        impact_reg.save_model()

    # ---- Comparative Summary ------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<35s} {'Accuracy':>10s} {'Macro-F1':>10s}")
    print("-" * 60)
    print(f"{'Event Flat (TF-IDF+LR+SMOTE)':<35s} {event_accuracy:>10.4f} {event_f1:>10.4f}")
    print(f"{'Event Hierarchical':<35s} {hier_accuracy:>10.4f} {hier_f1:>10.4f}")
    print(f"{'Event Ensemble (LR+Keywords)':<35s} {ens_event_acc:>10.4f} {ens_event_f1:>10.4f}")
    print(f"{'Stance (3-class)':<35s} {stance_accuracy:>10.4f} {stance_f1:>10.4f}")
    print(f"{'Stance Ensemble (LR+Keywords)':<35s} {ens_stance_acc:>10.4f} {ens_stance_f1:>10.4f}")
    if extra_results:
        print("-" * 60)
        for name, (acc, f1) in extra_results.items():
            label = f"Event {name}"
            print(f"{label:<35s} {acc:>10.4f} {f1:>10.4f}")
    print("-" * 60)
    print(f"{'Event 5-fold CV mean':<35s} {ev_acc:>10.4f} {ev_f1:>10.4f}")
    print(f"{'Stance 5-fold CV mean':<35s} {st_acc:>10.4f} {st_f1:>10.4f}")
    print("=" * 60)
    print(f"\nheadline_weight={headline_weight}  "
          f"(persisted to text_features_config.json — inference paths read this)")
    print("=" * 60)

    return event_clf, stance_det, impact_reg


# ---------------------------------------------------------------------------
# V3 — Relevance Classifier Training
# ---------------------------------------------------------------------------

def train_relevance_model(path: str = RELEVANCE_DATASET):
    """
    Stage 2 — Train the binary relevance classifier.

    Loads the full relevance dataset (~5,447 articles, both rel=0 and
    rel=1) exported by ``scripts/export_relevance_dataset.py``, splits
    80/20, fits a TF-IDF + Logistic Regression model, prints a
    classification report, and saves the model to disk.

    Returns
    -------
    RelevanceClassifierLR | None
        Trained classifier (None if training failed).
    """
    if not os.path.isfile(path):
        logger.error(
            "Relevance dataset not found: %s\n"
            "Run scripts/export_relevance_dataset.py first.", path,
        )
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d rows from %s", len(df), path)

    # Use the shared text builder so the relevance classifier sees the
    # same headline-weighted construction at train and inference time.
    df["headline"] = df["headline"].fillna("")
    df["clean_text"] = df["clean_text"].fillna("")
    df["text"] = build_input_texts(
        df["headline"].tolist(), df["clean_text"].tolist(),
    )

    df = df[df["text"] != ""].reset_index(drop=True)
    df = df.dropna(subset=["relevance"]).reset_index(drop=True)
    df["relevance"] = df["relevance"].astype(int)
    if len(df) < 50:
        logger.error("Need >= 50 rows to train relevance model. Got %d.", len(df))
        return None

    X = df["text"].tolist()
    y = df["relevance"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info("Train=%d  Test=%d  pos_rate=%.3f",
                len(X_train), len(X_test), sum(y_train) / max(len(y_train), 1))

    clf = RelevanceClassifierLR()
    if len(X_train) < 100:
        clf.vectorizer.set_params(min_df=1, max_df=1.0)
    clf.train(X_train, y_train)

    y_pred = clf.predict_batch(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RELEVANCE CLASSIFIER — Results")
    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["irrelevant", "relevant"], zero_division=0))

    clf.save_model()
    logger.info("Relevance classifier trained and saved.")
    return clf


# ---------------------------------------------------------------------------
# V3 — Sub-Score Regressors (materiality / linkage / time / credibility)
# ---------------------------------------------------------------------------

def train_sub_score_models(path: str = LABELED_DATASET):
    """
    V3 Stage 4b — Train one Ridge regressor per sub-score on the
    1,817-row gold dataset (relevance=1 only).

    The four trained regressors replace V2's Gemini grading at runtime;
    the deterministic weighted-sum formula in
    ``models/sub_score_regressor.py`` then computes ``impact_score``.
    """
    if not os.path.isfile(path):
        logger.error(
            "Gold dataset not found: %s\n"
            "Run scripts/export_gold_dataset.py first.", path,
        )
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    missing = [s for s in SUB_SCORES if s not in df.columns]
    if missing:
        logger.error("Gold dataset is missing sub-score columns: %s", missing)
        return None

    # Drop rows with any NULL sub-score
    before = len(df)
    df = df.dropna(subset=list(SUB_SCORES)).reset_index(drop=True)
    if before - len(df):
        logger.info("Dropped %d rows with missing sub-scores.", before - len(df))

    if len(df) < 50:
        logger.error("Need >= 50 rows to train sub-score regressors. Got %d.", len(df))
        return None

    # Use shared headline-weighted text builder so sub-scores see the same
    # input construction the rest of the pipeline uses at inference time.
    if "headline" not in df.columns:
        df["headline"] = ""
    df["headline"] = df["headline"].fillna("")
    X = build_input_texts(df["headline"].tolist(), df["clean_text"].tolist())
    targets = {s: df[s].astype(float).tolist() for s in SUB_SCORES}

    # Hold out 20% for evaluation
    indices = list(range(len(X)))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    X_train = [X[i] for i in train_idx]
    X_test  = [X[i] for i in test_idx]
    y_train = {s: [targets[s][i] for i in train_idx] for s in SUB_SCORES}
    y_test  = {s: [targets[s][i] for i in test_idx]  for s in SUB_SCORES}

    reg = SubScoreRegressor()
    if len(X_train) < 100:
        reg.vectorizer.set_params(min_df=1, max_df=1.0)
    reg.train(X_train, y_train)

    # Evaluate per sub-score
    print("\n" + "=" * 60)
    print("SUB-SCORE REGRESSORS — Results (held-out 20%)")
    print("=" * 60)
    preds = reg.predict_batch(X_test)
    for s in SUB_SCORES:
        y_t = y_test[s]
        y_p = [p[s] for p in preds]
        mae = mean_absolute_error(y_t, y_p)
        r2  = r2_score(y_t, y_p)
        print(f"  {s:18s}  MAE={mae:.4f}  R²={r2:.4f}")

    # Composite impact score MAE (using deterministic formula)
    from models.sub_score_regressor import compute_impact_score, WEIGHTS
    actual_impact = [
        compute_impact_score({s: y_test[s][i] for s in SUB_SCORES})
        for i in range(len(X_test))
    ]
    pred_impact = [p["impact_score"] for p in preds]
    print(f"\n  composite impact_score   MAE={mean_absolute_error(actual_impact, pred_impact):.4f}")
    print(f"  weights used: {WEIGHTS}")

    reg.save_model()
    logger.info("Sub-score regressors trained and saved.")
    return reg


# ---------------------------------------------------------------------------
# Test Prediction Function
# ---------------------------------------------------------------------------

def test_prediction(article_text: str) -> dict[str, str]:
    """
    Accept a new news article and predict its event category and stance.

    Loads the saved models from disk and returns both predictions.

    Parameters
    ----------
    article_text : str
        The cleaned text of a news article.

    Returns
    -------
    dict[str, str]
        Dictionary with keys ``"event_category"``, ``"stance"``,
        and ``"impact_score"``.

    Example
    -------
    >>> result = test_prediction("Apple reports record quarterly earnings")
    >>> print(result)
    {'event_category': 'Earnings', 'stance': 'bullish', 'impact_score': 0.85}
    """
    # Load saved models
    event_clf = EventClassifierLR()
    event_clf.load_model()

    stance_det = StanceDetectorLR()
    stance_det.load_model()

    impact_reg = ImpactRegressor()
    impact_reg.load_model()

    # Predict
    event_category = event_clf.predict_event(article_text)
    stance = stance_det.predict_stance(article_text)
    impact_score = impact_reg.predict(article_text)

    return {
        "event_category": event_category,
        "stance": stance,
        "impact_score": round(impact_score, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RNIA Enhanced Model Training")
    ap.add_argument("--no-expanded", action="store_true", help="Use gold-only dataset")
    ap.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    ap.add_argument(
        "--headline-weight", type=int, default=None,
        help="Headline repetition factor (default: reuse last-saved value from config; recommended 3)",
    )
    ap.add_argument(
        "--extra-event-model", choices=["xgb", "hybrid", "both"], default=None,
        help="Train an additional event classifier alongside LR for comparison",
    )
    args = ap.parse_args()

    result = train_and_evaluate(
        use_expanded=not args.no_expanded,
        use_smote=not args.no_smote,
        headline_weight=args.headline_weight,
        extra_event_model=args.extra_event_model,
    )

    if result:
        print("\n" + "=" * 60)
        print("TEST PREDICTION")
        print("=" * 60)

        sample_article = (
            "apple reported strong revenue growth in its quarterly earnings "
            "beating analyst expectations with record iphone sales"
        )
        print(f"\nInput: {sample_article}\n")

        prediction = test_prediction(sample_article)
        print(f"  Event Category : {prediction['event_category']}")
        print(f"  Stance         : {prediction['stance']}")
        print(f"  Impact Score   : {prediction['impact_score']}")
        print("=" * 60)

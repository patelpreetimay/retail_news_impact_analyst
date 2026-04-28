"""
run_optuna_search.py — Hyperparameter optimisation for event + stance models
=============================================================================

Re-tunes TF-IDF + Logistic Regression hyperparams via Optuna with a
wider search space and more trials than the original cached
``best_hyperparams.json``. Writes the new best params back to disk so
subsequent ``train_models.py`` runs pick them up automatically.

Search space (per model)
------------------------
TF-IDF:
    - ngram_max ∈ {1, 2, 3}
    - max_features ∈ {3000, 5000, 7500, 10000, 15000}
    - min_df ∈ {1, 2, 3, 4, 5}
    - max_df ∈ U(0.85, 0.99)
    - sublinear_tf ∈ {True, False}

Logistic Regression:
    - C ∈ LogU(0.05, 20.0)
    - class_weight ∈ {"balanced", None}

Objective
---------
5-fold stratified CV macro-F1 (event uses SMOTE on training folds; stance does not).

Usage
-----
    python scripts/run_optuna_search.py                     # 200 trials each
    python scripts/run_optuna_search.py --n-trials 50       # quick run
    python scripts/run_optuna_search.py --target event      # only event
    python scripts/run_optuna_search.py --target stance     # only stance
    python scripts/run_optuna_search.py --headline-weight 3 # weight headlines 3x
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from taxonomy.event_taxonomy import EVENT_COLLAPSE
from utils.text_features import build_input_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LABELED_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv"
)
EXPANDED_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled_expanded.csv"
)
HYPERPARAMS_PATH = os.path.join(
    PROJECT_ROOT, "models", "saved_models", "best_hyperparams.json"
)

STANCE_COLLAPSE = {"mixed": "neutral"}
RANDOM_STATE = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(use_expanded: bool, headline_weight: int) -> pd.DataFrame:
    path = EXPANDED_DATASET if (use_expanded and os.path.isfile(EXPANDED_DATASET)) else LABELED_DATASET
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d rows from %s", len(df), path)

    df = df.dropna(subset=["clean_text", "event_type", "stance"]).reset_index(drop=True)
    df["stance"] = df["stance"].replace(STANCE_COLLAPSE)
    df["event_type"] = df["event_type"].replace(EVENT_COLLAPSE)
    df = df[df["event_type"] != "Unclassified"].reset_index(drop=True)

    # Build the model input text — uses headline weighting if requested
    df["headline"] = df.get("headline", "").fillna("") if "headline" in df.columns else ""
    if "headline" not in df.columns:
        df["headline"] = ""
    df["clean_text"] = df["clean_text"].fillna("")
    df["input_text"] = build_input_texts(
        df["headline"].tolist(), df["clean_text"].tolist(), weight=headline_weight,
    )
    df = df[df["input_text"] != ""].reset_index(drop=True)

    logger.info("After cleanup: %d rows  (headline_weight=%d)", len(df), headline_weight)
    logger.info("Event distribution: %s", dict(Counter(df["event_type"])))
    logger.info("Stance distribution: %s", dict(Counter(df["stance"])))
    return df


# ---------------------------------------------------------------------------
# SMOTE (sparse TF-IDF feature space)
# ---------------------------------------------------------------------------

def maybe_smote(X_tfidf, y):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_tfidf, y

    counts = Counter(y)
    if min(counts.values()) < 2:
        return X_tfidf, y
    k = min(5, min(counts.values()) - 1)
    smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    return smote.fit_resample(X_tfidf, y)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(X: list[str], y: list[str], use_smote: bool):
    """Return an Optuna objective that 5-fold-CVs TF-IDF + LR."""

    def objective(trial):
        ngram_max = trial.suggest_categorical("ngram_max", [1, 2, 3])
        max_features = trial.suggest_categorical("max_features", [3000, 5000, 7500, 10000, 15000])
        min_df = trial.suggest_int("min_df", 1, 5)
        max_df = trial.suggest_float("max_df", 0.85, 0.99)
        sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])
        C = trial.suggest_float("C", 0.05, 20.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        f1s = []
        for train_idx, test_idx in skf.split(X, y):
            X_train = [X[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_train = [y[i] for i in train_idx]
            y_test = [y[i] for i in test_idx]

            vec = TfidfVectorizer(
                ngram_range=(1, ngram_max),
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
                stop_words="english",
            )
            X_train_tfidf = vec.fit_transform(X_train)
            X_test_tfidf = vec.transform(X_test)

            X_tr, y_tr = (X_train_tfidf, y_train)
            if use_smote:
                X_tr, y_tr = maybe_smote(X_tr, y_tr)

            clf = LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_test_tfidf)
            f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

        return float(np.mean(f1s))

    return objective


# ---------------------------------------------------------------------------
# Optuna driver
# ---------------------------------------------------------------------------

def run_search(target: str, X: list[str], y: list[str],
               n_trials: int, use_smote: bool, timeout: int | None) -> dict:
    try:
        import optuna
    except ImportError as e:
        raise SystemExit("optuna not installed. Run:  pip install optuna") from e

    # Quieten Optuna's default per-trial output but keep WARNING level for issues
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"rnia_{target}_search")

    logger.info("Starting Optuna search for '%s' (%d trials, SMOTE=%s)...",
                target, n_trials, use_smote)
    logger.info("This is NOT stuck — Optuna is running silently. Each trial does")
    logger.info("5-fold CV (5 model fits). Watch progress below; Ctrl+C is safe.")

    # Periodic progress callback so you SEE it running
    log_every = max(1, n_trials // 20)  # ~20 progress lines total

    def progress_cb(study_, trial):
        n = trial.number + 1
        if n == 1 or n % log_every == 0 or n == n_trials:
            logger.info("  Trial %3d/%d  this F1=%.4f  best so far=%.4f",
                        n, n_trials, trial.value or 0.0, study_.best_value)

    try:
        study.optimize(
            make_objective(X, y, use_smote=use_smote),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,    # tqdm-style live bar
            callbacks=[progress_cb],
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user — using best params from completed trials.")

    if not study.best_trial:
        raise RuntimeError(f"No completed trials for '{target}'. Cannot extract best params.")

    logger.info("Best CV macro-F1 for '%s': %.4f (after %d trials)",
                target, study.best_value, len(study.trials))
    logger.info("Best params: %s", study.best_params)

    return {
        "best_params": study.best_params,
        "best_macro_f1": round(float(study.best_value), 4),
        "n_trials": len(study.trials),
    }


def write_results(results: dict, headline_weight: int):
    """Update best_hyperparams.json with new search results, preserving any unchanged sections."""
    existing = {}
    if os.path.isfile(HYPERPARAMS_PATH):
        try:
            with open(HYPERPARAMS_PATH) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}

    existing.update(results)
    existing["text_features"] = {"headline_weight": int(headline_weight)}

    os.makedirs(os.path.dirname(HYPERPARAMS_PATH), exist_ok=True)
    with open(HYPERPARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    logger.info("Wrote best_hyperparams.json")
    print("\n--- best_hyperparams.json ---")
    print(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Optuna hyperparam search for RNIA")
    ap.add_argument("--n-trials", type=int, default=200,
                    help="Trials per target (default 200)")
    ap.add_argument("--target", choices=["event", "stance", "both"], default="both",
                    help="Which model(s) to tune")
    ap.add_argument("--no-expanded", action="store_true",
                    help="Use gold-only dataset")
    ap.add_argument("--no-smote", action="store_true",
                    help="Disable SMOTE on the event objective")
    ap.add_argument("--headline-weight", type=int, default=3,
                    help="Headline repetition factor for input text construction (default 3)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Per-target wall-clock timeout in seconds (optional)")
    args = ap.parse_args()

    df = load_data(use_expanded=not args.no_expanded,
                   headline_weight=args.headline_weight)

    X = df["input_text"].tolist()

    results: dict = {}

    if args.target in ("event", "both"):
        y_event = df["event_type"].tolist()
        results["event"] = run_search(
            target="event", X=X, y=y_event,
            n_trials=args.n_trials, use_smote=not args.no_smote,
            timeout=args.timeout,
        )

    if args.target in ("stance", "both"):
        y_stance = df["stance"].tolist()
        results["stance"] = run_search(
            target="stance", X=X, y=y_stance,
            n_trials=args.n_trials, use_smote=False,    # stance never uses SMOTE
            timeout=args.timeout,
        )

    write_results(results, headline_weight=args.headline_weight)
    print("\nDONE — re-run training (`python main.py` → option 3) "
          "to pick up the new hyperparams.")


if __name__ == "__main__":
    main()

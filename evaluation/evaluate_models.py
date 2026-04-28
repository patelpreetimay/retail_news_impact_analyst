"""
evaluate_models.py — Model Evaluation and Performance Analysis for RNIA
========================================================================

Loads the labeled dataset and trained models, splits the data 80/20,
evaluates the event classifier, stance detector, and the four sub-score
regressors. Generates confusion matrices for the classifiers and
scatter plots (actual vs predicted) for the regressors. Saves all results.

Inputs:
    data/labeled_dataset/financial_news_labeled.csv
    models/saved_models/event_classifier.pkl
    models/saved_models/stance_detector.pkl
    models/saved_models/tfidf_vectorizer.pkl
    models/saved_models/subscore_*.pkl  (4 regressors)
    models/saved_models/tfidf_subscore_vectorizer.pkl

Outputs:
    data/evaluation_results/model_performance.csv             (classifiers)
    data/evaluation_results/subscore_performance.csv          (regressors)
    data/evaluation_results/event_confusion_matrix.png
    data/evaluation_results/stance_confusion_matrix.png
    data/evaluation_results/subscore_scatter.png              (4-panel grid)
    data/evaluation_results/impact_score_scatter.png          (composed score)

Usage:
    python evaluation/evaluate_models.py
"""

import os
import sys
import logging
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.event_classifier import EventClassifierLR
from models.stance_detector import StanceDetectorLR
from models.sub_score_regressor import (
    SubScoreRegressor,
    SUB_SCORES,
    WEIGHTS,
    compute_impact_score,
)
from models.train_models import STANCE_COLLAPSE
from taxonomy.event_taxonomy import EVENT_COLLAPSE
from utils.text_features import build_input_texts, get_headline_weight

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LABELED_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation_results")
PERFORMANCE_CSV = os.path.join(RESULTS_DIR, "model_performance.csv")
SUBSCORE_PERFORMANCE_CSV = os.path.join(RESULTS_DIR, "subscore_performance.csv")
EVENT_CM_PNG = os.path.join(RESULTS_DIR, "event_confusion_matrix.png")
STANCE_CM_PNG = os.path.join(RESULTS_DIR, "stance_confusion_matrix.png")
SUBSCORE_SCATTER_PNG = os.path.join(RESULTS_DIR, "subscore_scatter.png")
IMPACT_SCATTER_PNG = os.path.join(RESULTS_DIR, "impact_score_scatter.png")

TEST_SIZE = 0.20       # 20% testing, 80% training
RANDOM_STATE = 42      # Reproducibility — must match trainer + finbert_comparison

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# STEP 2 — LOAD DATA AND MODELS
# ===========================================================================

def load_labeled_data(path: str = LABELED_DATASET) -> pd.DataFrame:
    """Load the labeled financial news dataset."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Labeled dataset not found: {path}\n"
            "Run annotation/annotation_tool.py to label articles first."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d labeled articles from: %s", len(df), path)
    return df


def load_trained_models():
    """Load classifiers + regressor stack from disk. Sub-score regressor is optional."""
    event_clf = EventClassifierLR()
    event_clf.load_model()
    logger.info("Event classifier loaded.")

    stance_det = StanceDetectorLR()
    stance_det.load_model()
    logger.info("Stance detector loaded.")

    sub_reg = None
    try:
        sub_reg = SubScoreRegressor()
        sub_reg.load_model()
        logger.info("Sub-score regressor loaded.")
    except FileNotFoundError as e:
        logger.warning("Sub-score regressor not available — skipping regression eval. (%s)", e)
        sub_reg = None

    return event_clf, stance_det, sub_reg


# ===========================================================================
# STEP 5 — CONFUSION MATRIX PLOTTING
# ===========================================================================

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 1.0)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to: %s", save_path)


# ===========================================================================
# STEP 5b — REGRESSION SCATTER PLOTS (sub-scores)
# ===========================================================================

def plot_subscore_scatter_grid(metrics_per_score, save_path):
    """
    4-panel grid (one panel per sub-score) of actual vs predicted values.
    Each subplot also shows MAE/R² in its title.
    """
    n = len(SUB_SCORES)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 9))
    axes = axes.flatten()

    for ax, name in zip(axes, SUB_SCORES):
        m = metrics_per_score[name]
        y_true = m["y_true"]
        y_pred = m["y_pred"]
        ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#2E86AB", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(
            f"{name} (weight={WEIGHTS[name]:.1f}) — MAE={m['mae']:.4f}, R²={m['r2']:.3f}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.grid(alpha=0.25)

    # Hide any unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Sub-score Regressors — Actual vs Predicted",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Sub-score scatter saved to: %s", save_path)


def plot_impact_scatter(y_true, y_pred, mae, r2, save_path):
    """Scatter of actual vs computed impact score (composed from predicted sub-scores)."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=22, color="#A23B72", edgecolors="none")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Actual impact score", fontsize=11)
    ax.set_ylabel("Predicted impact score (0.4·M + 0.3·L + 0.2·T + 0.1·C)", fontsize=11)
    ax.set_title(
        f"Impact Score — Actual vs Predicted (MAE={mae:.4f}, R²={r2:.3f})",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Impact-score scatter saved to: %s", save_path)


# ===========================================================================
# STEP 6 — SUB-SCORE EVALUATION
# ===========================================================================

def evaluate_sub_scores(df_test, sub_reg):
    """
    Evaluate the four sub-score Ridge regressors and the composed impact score
    on the held-out test slice. Returns a dict of per-sub-score metrics +
    one entry for the composed impact score.
    """
    needed_cols = list(SUB_SCORES) + ["impact_score"]
    df_eval = df_test.dropna(subset=["clean_text"] + needed_cols).reset_index(drop=True)
    dropped = len(df_test) - len(df_eval)
    if dropped:
        logger.warning("Dropped %d test rows missing sub-score targets. "
                       "Evaluating sub-scores on %d rows.", dropped, len(df_eval))

    if len(df_eval) == 0:
        logger.warning("No test rows have all sub-score targets. Skipping regression eval.")
        return None

    # Match training-time text construction (headline weighting) for sub-scores too
    if "headline" not in df_eval.columns:
        df_eval["headline"] = ""
    df_eval["headline"] = df_eval["headline"].fillna("")
    df_eval["clean_text"] = df_eval["clean_text"].fillna("")
    X_test = build_input_texts(
        df_eval["headline"].tolist(),
        df_eval["clean_text"].tolist(),
    )
    predictions = sub_reg.predict_batch(X_test)  # list[dict]

    metrics = {}
    print("\n" + "=" * 60)
    print("SUB-SCORE REGRESSORS — Evaluation (test set)")
    print("=" * 60)
    for name in SUB_SCORES:
        y_true = df_eval[name].astype(float).clip(0.0, 1.0).tolist()
        y_pred = [p[name] for p in predictions]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        metrics[name] = {
            "y_true": y_true, "y_pred": y_pred,
            "mae": mae, "rmse": rmse, "r2": r2,
        }
        print(f"  {name:18s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:+.4f}  weight={WEIGHTS[name]:.1f}")

    # Composed impact score from predicted sub-scores
    y_impact_true = df_eval["impact_score"].astype(float).clip(0.0, 1.0).tolist()
    y_impact_pred = [compute_impact_score(p) for p in predictions]
    impact_mae = mean_absolute_error(y_impact_true, y_impact_pred)
    impact_rmse = math.sqrt(mean_squared_error(y_impact_true, y_impact_pred))
    impact_r2 = r2_score(y_impact_true, y_impact_pred)
    metrics["impact_score"] = {
        "y_true": y_impact_true, "y_pred": y_impact_pred,
        "mae": impact_mae, "rmse": impact_rmse, "r2": impact_r2,
    }
    print(f"  {'impact_score':18s}  MAE={impact_mae:.4f}  RMSE={impact_rmse:.4f}  R²={impact_r2:+.4f}  (composed)")

    return metrics


def save_subscore_csv(metrics, n_test, save_path):
    """Write a tidy CSV — one row per sub-score plus the composed impact score."""
    rows = []
    for name in list(SUB_SCORES) + ["impact_score"]:
        m = metrics[name]
        rows.append({
            "score_name": name,
            "weight": WEIGHTS.get(name, 1.0),
            "mae": round(m["mae"], 4),
            "rmse": round(m["rmse"], 4),
            "r2": round(m["r2"], 4),
            "n_test_samples": n_test,
        })
    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info("Sub-score performance saved to: %s", save_path)
    return df_out


# ===========================================================================
# STEP 4 & 6 — EVALUATE AND SAVE
# ===========================================================================

def evaluate_models():
    """
    Full evaluation pipeline:
        1. Load labeled dataset
        2. Load trained models (event clf, stance det, sub-score regressor)
        3. Split data 80/20 by index (preserves dataframe alignment)
        4. Predict with all models
        5. Compute classifier metrics + regressor metrics
        6. Print classification reports
        7. Generate confusion matrices + sub-score scatter plots
        8. Save metrics CSVs
    """
    logger.info("=" * 60)
    logger.info("RNIA — Model Evaluation & Performance Analysis")
    logger.info("=" * 60)

    # ---- Step 2: Load data and models ----------------------------------------
    df = load_labeled_data()
    df = df.dropna(subset=["clean_text", "event_type", "stance"]).reset_index(drop=True)
    logger.info("Articles after dropping NaN: %d", len(df))

    # Mirror the training-time label space so y_true and the model's predictions
    # share a vocabulary. Without this, raw 9-class CSV labels (e.g.
    # "Macroeconomic_Geopolitical") are scored against the model's collapsed
    # 7-class output ("Market_Movement"), which auto-fails ~66% of rows.
    df["event_type"] = df["event_type"].replace(EVENT_COLLAPSE)
    df["stance"]     = df["stance"].replace(STANCE_COLLAPSE)
    df = df[df["event_type"] != "Unclassified"].reset_index(drop=True)
    logger.info("Articles after collapse + dropping Unclassified: %d", len(df))

    if len(df) < 2:
        logger.error("Need at least 2 labeled articles. Found %d.", len(df))
        return

    event_clf, stance_det, sub_reg = load_trained_models()

    # ---- Step 3: Index-based split (keeps dataframe rows aligned) -----------
    small_dataset = len(df) < 10
    if small_dataset:
        logger.warning(
            "Small dataset (%d samples). Evaluating on ALL data. "
            "For proper evaluation, label more articles first.",
            len(df),
        )
        df_test = df.copy()
        eval_label = "training data (small dataset)"
    else:
        all_idx = np.arange(len(df))
        _, test_idx = train_test_split(
            all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        df_test = df.iloc[test_idx].reset_index(drop=True)
        eval_label = "test set (20%)"

    # CRITICAL: build eval input the SAME way training did (headline weighting),
    # so the models see the input distribution they were trained on. Without
    # this, models trained with headline_weight=3 will be evaluated on raw
    # clean_text only and look ~2-4% worse than their true performance.
    if "headline" not in df_test.columns:
        df_test["headline"] = ""
    df_test["headline"] = df_test["headline"].fillna("")
    df_test["clean_text"] = df_test["clean_text"].fillna("")

    weight = get_headline_weight()
    X_test = build_input_texts(
        df_test["headline"].tolist(),
        df_test["clean_text"].tolist(),
        weight=weight,
    )
    y_event_test = df_test["event_type"].tolist()
    y_stance_test = df_test["stance"].tolist()

    logger.info("Evaluating on %d samples (%s, headline_weight=%d)",
                len(X_test), eval_label, weight)

    # ---- Step 4: Predict and evaluate ----------------------------------------
    # --- Event Classifier ---
    y_event_pred = [event_clf.predict_event(text) for text in X_test]

    event_accuracy = accuracy_score(y_event_test, y_event_pred)
    event_precision = precision_score(y_event_test, y_event_pred, average="weighted", zero_division=0)
    event_recall = recall_score(y_event_test, y_event_pred, average="weighted", zero_division=0)
    event_f1 = f1_score(y_event_test, y_event_pred, average="weighted", zero_division=0)

    # --- Stance Detector ---
    y_stance_pred = [stance_det.predict_stance(text) for text in X_test]

    stance_accuracy = accuracy_score(y_stance_test, y_stance_pred)
    stance_precision = precision_score(y_stance_test, y_stance_pred, average="weighted", zero_division=0)
    stance_recall = recall_score(y_stance_test, y_stance_pred, average="weighted", zero_division=0)
    stance_f1 = f1_score(y_stance_test, y_stance_pred, average="weighted", zero_division=0)

    # --- Compact side-by-side summary FIRST (the table you want at the top) ---
    print("\n" + "=" * 72)
    print(f"CLASSIFIER METRICS — {eval_label}")
    print("=" * 72)
    print(f"{'Model':<22s} {'Accuracy':>10s} {'Precision':>11s} {'Recall':>10s} {'F1':>10s}")
    print("-" * 72)
    print(f"{'Event Classifier':<22s} {event_accuracy:>10.4f} {event_precision:>11.4f} "
          f"{event_recall:>10.4f} {event_f1:>10.4f}")
    print(f"{'Stance Detector':<22s} {stance_accuracy:>10.4f} {stance_precision:>11.4f} "
          f"{stance_recall:>10.4f} {stance_f1:>10.4f}")
    print("=" * 72)

    # --- Detailed per-class breakdowns (kept for the report / viva backup) ---
    print("\n" + "-" * 60)
    print(f"EVENT CLASSIFIER — Per-class report")
    print("-" * 60)
    print(classification_report(y_event_test, y_event_pred, zero_division=0))

    print("-" * 60)
    print(f"STANCE DETECTOR — Per-class report")
    print("-" * 60)
    print(classification_report(y_stance_test, y_stance_pred, zero_division=0))

    # ---- Step 5 & 7: Confusion matrices and plots ----------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    event_labels = sorted(set(y_event_test + y_event_pred))
    plot_confusion_matrix(
        y_true=y_event_test, y_pred=y_event_pred,
        labels=event_labels,
        title="Event Classifier — Confusion Matrix",
        save_path=EVENT_CM_PNG,
    )

    stance_labels = sorted(set(y_stance_test + y_stance_pred))
    plot_confusion_matrix(
        y_true=y_stance_test, y_pred=y_stance_pred,
        labels=stance_labels,
        title="Stance Detector — Confusion Matrix",
        save_path=STANCE_CM_PNG,
    )

    # ---- Step 6a: Save classifier performance summary CSV --------------------
    performance_data = pd.DataFrame([
        {
            "model_name": "Event Classifier",
            "accuracy": round(event_accuracy, 4),
            "precision": round(event_precision, 4),
            "recall": round(event_recall, 4),
            "f1_score": round(event_f1, 4),
        },
        {
            "model_name": "Stance Detector",
            "accuracy": round(stance_accuracy, 4),
            "precision": round(stance_precision, 4),
            "recall": round(stance_recall, 4),
            "f1_score": round(stance_f1, 4),
        },
    ])

    performance_data.to_csv(PERFORMANCE_CSV, index=False, encoding="utf-8-sig")
    logger.info("Performance summary saved to: %s", PERFORMANCE_CSV)

    # ---- Step 6b: Sub-score regressor evaluation -----------------------------
    subscore_summary = None
    if sub_reg is not None:
        subscore_metrics = evaluate_sub_scores(df_test, sub_reg)
        if subscore_metrics is not None:
            # Save scatter grid (4 sub-scores) + impact-score scatter
            plot_subscore_scatter_grid(subscore_metrics, SUBSCORE_SCATTER_PNG)
            impact_m = subscore_metrics["impact_score"]
            plot_impact_scatter(
                impact_m["y_true"], impact_m["y_pred"],
                impact_m["mae"], impact_m["r2"],
                IMPACT_SCATTER_PNG,
            )
            # Save subscore CSV
            subscore_summary = save_subscore_csv(
                subscore_metrics,
                n_test=len(subscore_metrics[SUB_SCORES[0]]["y_true"]),
                save_path=SUBSCORE_PERFORMANCE_CSV,
            )

    # ---- Final summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  Classifier performance CSV : {PERFORMANCE_CSV}")
    print(f"  Event confusion matrix     : {EVENT_CM_PNG}")
    print(f"  Stance confusion matrix    : {STANCE_CM_PNG}")
    if subscore_summary is not None:
        print(f"  Sub-score performance CSV  : {SUBSCORE_PERFORMANCE_CSV}")
        print(f"  Sub-score scatter grid     : {SUBSCORE_SCATTER_PNG}")
        print(f"  Impact-score scatter       : {IMPACT_SCATTER_PNG}")
    print("\nClassifier Performance Summary:")
    print(performance_data.to_string(index=False))
    if subscore_summary is not None:
        print("\nSub-score Regressor Performance Summary:")
        print(subscore_summary.to_string(index=False))
    print("=" * 60)

    return performance_data


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    evaluate_models()

"""
evaluate_models.py — Model Evaluation and Performance Analysis for RNIA
========================================================================

Loads the labeled dataset and trained models, splits the data 80/20,
evaluates both the event classifier and stance detector, generates
confusion matrices, and saves all results.

Inputs:
    data/labeled_dataset/financial_news_labeled.csv
    models/saved_models/event_classifier.pkl
    models/saved_models/stance_detector.pkl
    models/saved_models/tfidf_vectorizer.pkl

Outputs:
    data/evaluation_results/model_performance.csv
    data/evaluation_results/event_confusion_matrix.png
    data/evaluation_results/stance_confusion_matrix.png

Usage:
    python evaluation/evaluate_models.py
"""

import os
import sys
import logging

import joblib
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
)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.event_classifier import EventClassifierLR
from models.stance_detector import StanceDetectorLR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LABELED_DATASET = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation_results")
PERFORMANCE_CSV = os.path.join(RESULTS_DIR, "model_performance.csv")
EVENT_CM_PNG = os.path.join(RESULTS_DIR, "event_confusion_matrix.png")
STANCE_CM_PNG = os.path.join(RESULTS_DIR, "stance_confusion_matrix.png")

TEST_SIZE = 0.20       # 20% testing, 80% training
RANDOM_STATE = 42      # Reproducibility

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
    """
    Load the labeled financial news dataset.

    Parameters
    ----------
    path : str
        Path to the labeled CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including clean_text, event_type, stance.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Labeled dataset not found: {path}\n"
            "Run annotation/annotation_tool.py to label articles first."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d labeled articles from: %s", len(df), path)
    return df


def load_trained_models():
    """
    Load trained event classifier and stance detector from disk.

    Returns
    -------
    tuple[EventClassifierLR, StanceDetectorLR]
        The loaded event classifier and stance detector.
    """
    # Load event classifier
    event_clf = EventClassifierLR()
    event_clf.load_model()
    logger.info("Event classifier loaded.")

    # Load stance detector
    stance_det = StanceDetectorLR()
    stance_det.load_model()
    logger.info("Stance detector loaded.")

    return event_clf, stance_det


# ===========================================================================
# STEP 5 — CONFUSION MATRIX PLOTTING
# ===========================================================================

def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    labels: list,
    title: str,
    save_path: str,
) -> None:
    """
    Generate and save a confusion matrix plot.

    Parameters
    ----------
    y_true : list
        True labels.
    y_pred : list
        Predicted labels.
    labels : list
        List of unique class labels (for axis ordering).
    title : str
        Title for the plot.
    save_path : str
        File path to save the figure (.png).
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create figure with a clean style
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 1.0)))

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=True,
    )

    # Styling
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to: %s", save_path)


# ===========================================================================
# STEP 4 & 6 — EVALUATE AND SAVE
# ===========================================================================

def evaluate_models():
    """
    Full evaluation pipeline:
        1. Load labeled dataset
        2. Load trained models
        3. Split data 80/20 (or use all data if < 10 samples)
        4. Predict with both models
        5. Compute Accuracy, Precision, Recall, F1 for each
        6. Print classification reports
        7. Generate and save confusion matrix plots
        8. Save summary metrics to CSV
    """
    logger.info("=" * 60)
    logger.info("RNIA — Model Evaluation & Performance Analysis")
    logger.info("=" * 60)

    # ---- Step 2: Load data and models ----------------------------------------
    df = load_labeled_data()
    df = df.dropna(subset=["clean_text", "event_type", "stance"]).reset_index(drop=True)
    logger.info("Articles after dropping NaN: %d", len(df))

    if len(df) < 2:
        logger.error("Need at least 2 labeled articles. Found %d.", len(df))
        return

    event_clf, stance_det = load_trained_models()

    # Features and labels
    X = df["clean_text"].tolist()
    y_event_true = df["event_type"].tolist()
    y_stance_true = df["stance"].tolist()

    # ---- Step 3: Data split --------------------------------------------------
    small_dataset = len(df) < 10

    if small_dataset:
        logger.warning(
            "Small dataset (%d samples). Evaluating on ALL data. "
            "For proper evaluation, label more articles via "
            "annotation/annotation_tool.py.",
            len(df),
        )
        X_test = X
        y_event_test = y_event_true
        y_stance_test = y_stance_true
        eval_label = "training data (small dataset)"
    else:
        _, X_test, _, y_event_test, _, y_stance_test = train_test_split(
            X, y_event_true, y_stance_true,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        eval_label = "test set (20%)"

    logger.info("Evaluating on %d samples (%s)", len(X_test), eval_label)

    # ---- Step 4: Predict and evaluate ----------------------------------------
    # --- Event Classifier ---
    y_event_pred = [event_clf.predict_event(text) for text in X_test]

    event_accuracy = accuracy_score(y_event_test, y_event_pred)
    event_precision = precision_score(y_event_test, y_event_pred, average="weighted", zero_division=0)
    event_recall = recall_score(y_event_test, y_event_pred, average="weighted", zero_division=0)
    event_f1 = f1_score(y_event_test, y_event_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print(f"EVENT CLASSIFIER — Evaluation ({eval_label})")
    print("=" * 60)
    print(f"  Accuracy  : {event_accuracy:.4f}")
    print(f"  Precision : {event_precision:.4f}")
    print(f"  Recall    : {event_recall:.4f}")
    print(f"  F1 Score  : {event_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_event_test, y_event_pred, zero_division=0))

    # --- Stance Detector ---
    y_stance_pred = [stance_det.predict_stance(text) for text in X_test]

    stance_accuracy = accuracy_score(y_stance_test, y_stance_pred)
    stance_precision = precision_score(y_stance_test, y_stance_pred, average="weighted", zero_division=0)
    stance_recall = recall_score(y_stance_test, y_stance_pred, average="weighted", zero_division=0)
    stance_f1 = f1_score(y_stance_test, y_stance_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print(f"STANCE DETECTOR — Evaluation ({eval_label})")
    print("=" * 60)
    print(f"  Accuracy  : {stance_accuracy:.4f}")
    print(f"  Precision : {stance_precision:.4f}")
    print(f"  Recall    : {stance_recall:.4f}")
    print(f"  F1 Score  : {stance_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_stance_test, y_stance_pred, zero_division=0))

    # ---- Step 5 & 7: Confusion matrices and plots ----------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Event classifier confusion matrix
    event_labels = sorted(set(y_event_test + y_event_pred))
    plot_confusion_matrix(
        y_true=y_event_test,
        y_pred=y_event_pred,
        labels=event_labels,
        title="Event Classifier — Confusion Matrix",
        save_path=EVENT_CM_PNG,
    )

    # Stance detector confusion matrix
    stance_labels = sorted(set(y_stance_test + y_stance_pred))
    plot_confusion_matrix(
        y_true=y_stance_test,
        y_pred=y_stance_pred,
        labels=stance_labels,
        title="Stance Detector — Confusion Matrix",
        save_path=STANCE_CM_PNG,
    )

    # ---- Step 6: Save performance summary CSV --------------------------------
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

    # ---- Final summary -------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nSaved files:")
    print(f"  Performance CSV      : {PERFORMANCE_CSV}")
    print(f"  Event confusion matrix: {EVENT_CM_PNG}")
    print(f"  Stance confusion matrix: {STANCE_CM_PNG}")
    print("\nPerformance Summary:")
    print(performance_data.to_string(index=False))
    print("=" * 60)

    return performance_data


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    evaluate_models()

"""
train_models.py — Model Training Script for RNIA
=================================================

Loads the labeled dataset, splits it 80/20, trains both the event
classifier and stance detector, evaluates them, saves the trained
models to disk, and provides a test prediction function.

Input:
    data/labeled_dataset/financial_news_labeled.csv

Output Models (saved via joblib):
    models/saved_models/event_classifier.pkl
    models/saved_models/stance_detector.pkl
    models/saved_models/tfidf_vectorizer.pkl

Evaluation Metrics:
    Accuracy, Precision, Recall, F1 Score (via classification_report)
"""

import os
import sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# Project root on sys.path
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

TEST_SIZE = 0.20       # 20% testing, 80% training
RANDOM_STATE = 42      # Reproducibility

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


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
        DataFrame with columns: headline, clean_text, event_type, stance,
        source, timestamp, url.

    Raises
    ------
    FileNotFoundError
        If the labeled dataset file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Labeled dataset not found at:\n  {path}\n"
            "Please run annotation/annotation_tool.py to label articles first."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Loaded %d labeled articles from: %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def train_and_evaluate():
    """
    Full training pipeline:
        1. Load labeled dataset
        2. Split into 80% train / 20% test (if enough data)
        3. Train event classifier (TF-IDF + Logistic Regression)
        4. Train stance detector (TF-IDF + Logistic Regression)
        5. Evaluate both models and print classification reports
        6. Save trained models to models/saved_models/
    """
    # ---- Step 1: Load data --------------------------------------------------
    logger.info("=" * 60)
    logger.info("RNIA — Model Training Pipeline")
    logger.info("=" * 60)

    df = load_labeled_data()

    # Drop rows with missing text or labels
    df = df.dropna(subset=["clean_text", "event_type", "stance"]).reset_index(drop=True)
    logger.info("Articles after dropping NaN: %d", len(df))

    if len(df) < 2:
        logger.error("Need at least 2 labeled articles to train. Found %d.", len(df))
        return

    # Features and labels
    X = df["clean_text"].tolist()
    y_event = df["event_type"].tolist()
    y_stance = df["stance"].tolist()

    # ---- Step 2: Train/Test Split -------------------------------------------
    # For small datasets, train on ALL data and evaluate on training data.
    # For larger datasets (>= 10 samples), use proper 80/20 split.
    small_dataset = len(df) < 10

    if small_dataset:
        logger.warning(
            "Small dataset (%d samples). Training on ALL data; "
            "evaluation will be on training set. Label more articles "
            "for a proper train/test split.",
            len(df),
        )
        X_train = X
        X_test = X
        y_event_train = y_event
        y_event_test = y_event
        y_stance_train = y_stance
        y_stance_test = y_stance
    else:
        # Try stratified split; fall back to non-stratified if any
        # class has too few members for stratification.
        try:
            (
                X_train, X_test,
                y_event_train, y_event_test,
                y_stance_train, y_stance_test,
            ) = train_test_split(
                X, y_event, y_stance,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_event,
            )
        except ValueError:
            logger.warning(
                "Stratified split failed (some classes have < 2 samples). "
                "Falling back to non-stratified split."
            )
            (
                X_train, X_test,
                y_event_train, y_event_test,
                y_stance_train, y_stance_test,
            ) = train_test_split(
                X, y_event, y_stance,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

    logger.info("Train set size: %d", len(X_train))
    logger.info("Test  set size: %d", len(X_test))

    # ---- Step 3: Train Event Classifier -------------------------------------
    logger.info("")
    logger.info("-" * 60)
    logger.info("Training EVENT CLASSIFIER")
    logger.info("-" * 60)

    event_clf = EventClassifierLR()
    event_clf.train_event_classifier(X_train, y_event_train)

    # Evaluate
    y_event_pred = [event_clf.predict_event(text) for text in X_test]
    event_accuracy = accuracy_score(y_event_test, y_event_pred)

    eval_note = " (on training data — small dataset)" if small_dataset else ""
    print("\n" + "=" * 60)
    print(f"EVENT CLASSIFIER — Results{eval_note}")
    print("=" * 60)
    print(f"Accuracy: {event_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_event_test, y_event_pred, zero_division=0))

    # ---- Step 4: Train Stance Detector --------------------------------------
    logger.info("-" * 60)
    logger.info("Training STANCE DETECTOR")
    logger.info("-" * 60)

    stance_det = StanceDetectorLR()
    stance_det.train_stance_model(X_train, y_stance_train)

    # Evaluate
    y_stance_pred = [stance_det.predict_stance(text) for text in X_test]
    stance_accuracy = accuracy_score(y_stance_test, y_stance_pred)

    print("\n" + "=" * 60)
    print(f"STANCE DETECTOR — Results{eval_note}")
    print("=" * 60)
    print(f"Accuracy: {stance_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_stance_test, y_stance_pred, zero_division=0))

    # ---- Step 5: Save Models ------------------------------------------------
    logger.info("-" * 60)
    logger.info("Saving trained models...")
    logger.info("-" * 60)

    event_clf.save_model()
    stance_det.save_model()

    logger.info("")
    logger.info("=" * 60)
    logger.info("All models trained and saved successfully!")
    logger.info("=" * 60)

    return event_clf, stance_det



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
        Dictionary with keys ``"event_category"`` and ``"stance"``.

    Example
    -------
    >>> result = test_prediction("Apple reports record quarterly earnings")
    >>> print(result)
    {'event_category': 'earnings', 'stance': 'positive'}
    """
    # Load saved models
    event_clf = EventClassifierLR()
    event_clf.load_model()

    stance_det = StanceDetectorLR()
    stance_det.load_model()

    # Predict
    event_category = event_clf.predict_event(article_text)
    stance = stance_det.predict_stance(article_text)

    return {
        "event_category": event_category,
        "stance": stance,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Train and evaluate
    result = train_and_evaluate()

    if result:
        # Test prediction with a sample article
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
        print("=" * 60)

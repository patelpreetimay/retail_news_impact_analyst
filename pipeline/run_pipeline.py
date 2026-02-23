"""
run_pipeline.py — End-to-End NLP Pipeline for RNIA
===================================================

Reads the cleaned news dataset, runs event classification & stance detection
via zero-shot models, computes impact scores, and saves the final scored
dataset.

Input:
    data/processed_news/news_clean_dataset.csv

Output:
    data/results/news_with_impact_scores.csv

Output Columns:
    headline, clean_text, event_type, event_confidence,
    stance, stance_confidence, impact_score,
    source, timestamp, url
"""

import os
import sys
import time
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.event_classifier import EventClassifier
from models.stance_detector import StanceDetector
from models.impact_scorer import ImpactScorer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "results")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news_with_impact_scores.csv")

OUTPUT_COLUMNS = [
    "headline",
    "clean_text",
    "event_type",
    "event_confidence",
    "stance",
    "stance_confidence",
    "impact_score",
    "source",
    "timestamp",
    "url",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(input_path: str = INPUT_FILE, output_path: str = OUTPUT_FILE) -> pd.DataFrame:
    """
    Run the full NLP pipeline on the cleaned dataset.

    Steps:
        1. Load cleaned articles.
        2. Initialise models (event classifier, stance detector, impact scorer).
        3. Classify each article's event type and stance.
        4. Compute impact scores.
        5. Save final dataset to CSV.

    Parameters
    ----------
    input_path : str
        Path to the cleaned news CSV.
    output_path : str
        Path for the scored output CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with all output columns.
    """
    # --- Load dataset -------------------------------------------------------
    logger.info("Loading cleaned dataset from: %s", input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Cleaned dataset not found at: {input_path}\n"
            "Please run preprocessing/clean_text.py first."
        )
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info("Loaded %d articles.", len(df))

    # --- Initialise models ---------------------------------------------------
    logger.info("Initialising NLP models...")
    event_clf = EventClassifier()

    # Share the same underlying pipeline to save memory (~1.6 GB model)
    stance_det = StanceDetector(classifier=event_clf.pipe)

    scorer = ImpactScorer()
    logger.info("All models ready.\n")

    # --- Classify each article -----------------------------------------------
    event_types = []
    event_confs = []
    stances = []
    stance_confs = []

    total = len(df)
    start_time = time.time()

    for idx, row in df.iterrows():
        text = str(row.get("clean_text", ""))
        headline = str(row.get("headline", ""))

        # Combine headline + text for richer context
        combined = f"{headline}. {text}"

        # Event classification
        et, ec = event_clf.classify(combined)
        event_types.append(et)
        event_confs.append(ec)

        # Stance detection
        st, sc = stance_det.detect(combined)
        stances.append(st)
        stance_confs.append(sc)

        # Progress log every 10 articles
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d]  %.1f art/s  |  last → %s / %s",
                idx + 1,
                total,
                rate,
                et,
                st,
            )

    # --- Impact scores -------------------------------------------------------
    impact_scores = scorer.compute_batch(event_types, stances, event_confs, stance_confs)

    # --- Assemble output DataFrame -------------------------------------------
    df["event_type"] = event_types
    df["event_confidence"] = event_confs
    df["stance"] = stances
    df["stance_confidence"] = stance_confs
    df["impact_score"] = impact_scores

    df = df[OUTPUT_COLUMNS]

    # --- Save ----------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_time
    logger.info("\nPipeline complete in %.1f seconds.", elapsed)
    logger.info("Output saved to: %s", output_path)

    # --- Summary stats -------------------------------------------------------
    logger.info("-" * 50)
    logger.info("Event type distribution:")
    for et, count in df["event_type"].value_counts().items():
        logger.info("  %-25s %d", et, count)
    logger.info("")
    logger.info("Stance distribution:")
    for st, count in df["stance"].value_counts().items():
        logger.info("  %-12s %d", st, count)
    logger.info("")
    logger.info("Impact score stats:")
    logger.info("  Mean : %.2f", df["impact_score"].mean())
    logger.info("  Min  : %.2f", df["impact_score"].min())
    logger.info("  Max  : %.2f", df["impact_score"].max())
    logger.info("=" * 50)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()

"""
export_gold_dataset.py — Extract Gold Labels as Training Dataset
==================================================================

Reads the high-quality, Gemini-labelled articles from data/rnia.db
and exports them as the canonical training CSV for the ML pipeline.

The gold dataset is the relevance=1 + processing_status='done' subset
of v2_analyses (~1,800 rows), representing articles whose event_type,
stance, and impact sub-scores have been validated.

Output:
    data/labeled_dataset/financial_news_labeled.csv

Columns:
    headline, clean_text, event_type, stance, impact_score,
    materiality, market_linkage, time_sensitivity, credibility,
    source, timestamp, url, region, relevance
"""

import os
import sys
import sqlite3
import logging

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "rnia.db")
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stance normalisation: DB UPPERCASE → lowercase for ML training
# ---------------------------------------------------------------------------
STANCE_MAP = {
    "BULLISH":      "bullish",
    "BEARISH":      "bearish",
    "NEUTRAL":      "neutral",
    "MIXED":        "neutral",   # collapsed: too few samples (24) to learn
    "UNCLASSIFIED": "neutral",  # fold unclassified into neutral
}


def main():
    logger.info("=" * 60)
    logger.info("RNIA — Export Gold Labels as Training Dataset")
    logger.info("=" * 60)

    if not os.path.isfile(DB_PATH):
        logger.error("Database not found: %s", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Pull every relevant, done article + its analysis
    sql = """
        SELECT
            a.article_id,
            a.headline,
            a.clean_text,
            a.source,
            a.region,
            a.published_at  AS timestamp,
            a.url,
            v.event_type,
            v.stance,
            v.impact_score,
            v.materiality,
            v.market_linkage,
            v.time_sensitivity,
            v.credibility
        FROM articles a
        JOIN v2_analyses v ON v.article_id = a.article_id
        WHERE v.processing_status = 'done'
          AND v.relevance = 1
        ORDER BY a.article_id
    """

    df = pd.read_sql_query(sql, conn)
    conn.close()

    logger.info("Loaded %d relevant articles from data/rnia.db", len(df))

    # --- Normalise stance labels ---
    df["stance"] = df["stance"].map(STANCE_MAP).fillna("neutral")

    # --- Drop rows with missing text ---
    before = len(df)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d articles with empty clean_text", dropped)

    # --- Add relevance column (all RELEVANT by definition) ---
    df["relevance"] = "RELEVANT"

    # --- Reorder columns ---
    column_order = [
        "headline", "clean_text", "event_type", "stance",
        "impact_score", "materiality", "market_linkage",
        "time_sensitivity", "credibility",
        "source", "timestamp", "url", "region", "relevance",
    ]
    df = df[[c for c in column_order if c in df.columns]]

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    logger.info("Saved %d articles to: %s", len(df), OUTPUT_PATH)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("GOLD DATASET EXPORT — Summary")
    print("=" * 60)
    print(f"  Total articles exported : {len(df)}")

    print("\n  Event type distribution:")
    for et, count in df["event_type"].value_counts().items():
        print(f"    {et:40s} {count}")

    print("\n  Stance distribution:")
    for st, count in df["stance"].value_counts().items():
        print(f"    {st:15s} {count}")

    print(f"\n  Impact score range: {df['impact_score'].min():.2f} – {df['impact_score'].max():.2f}")
    print(f"  Impact score mean : {df['impact_score'].mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

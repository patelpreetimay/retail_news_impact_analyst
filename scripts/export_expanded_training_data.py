"""
export_expanded_training_data.py — Expanded Training Dataset with Silver Labels
================================================================================

Extends the gold export to include needs_escalation and failed articles
with relevance=1 that have Gemini-assigned labels.  These "silver" rows
add ~20% more training data.

Output:
    data/labeled_dataset/financial_news_labeled_expanded.csv

Columns (same as gold + label_source):
    headline, clean_text, event_type, stance, impact_score,
    materiality, market_linkage, time_sensitivity, credibility,
    source, timestamp, url, region, relevance, label_source
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
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled_expanded.csv"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stance normalisation: V2 UPPERCASE → lowercase, 4→3 collapse
# ---------------------------------------------------------------------------
STANCE_MAP = {
    "BULLISH":      "bullish",
    "BEARISH":      "bearish",
    "NEUTRAL":      "neutral",
    "MIXED":        "neutral",   # collapsed: too few samples
    "UNCLASSIFIED": "neutral",   # fold unclassified into neutral
}


def main():
    logger.info("=" * 60)
    logger.info("RNIA — Export EXPANDED Training Dataset (Gold + Silver)")
    logger.info("=" * 60)

    if not os.path.isfile(DB_PATH):
        logger.error("V2 database not found: %s", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Pull gold articles (done + relevant) AND silver (needs_escalation/failed + relevant)
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
            v.credibility,
            v.processing_status
        FROM articles a
        JOIN v2_analyses v ON v.article_id = a.article_id
        WHERE v.relevance = 1
          AND v.event_type IS NOT NULL
          AND v.stance IS NOT NULL
          AND v.processing_status IN ('done', 'needs_escalation', 'failed')
        ORDER BY a.article_id
    """

    df = pd.read_sql_query(sql, conn)
    conn.close()

    logger.info("Loaded %d articles from data/rnia.db", len(df))

    # --- Label source ---
    df["label_source"] = df["processing_status"].map(
        lambda s: "gold" if s == "done" else "silver"
    )
    gold_count = (df["label_source"] == "gold").sum()
    silver_count = (df["label_source"] == "silver").sum()
    logger.info("  Gold: %d,  Silver: %d", gold_count, silver_count)

    # --- Normalise stance labels (4→3 collapse) ---
    df["stance"] = df["stance"].map(STANCE_MAP).fillna("neutral")

    # --- Drop rows with missing text ---
    before = len(df)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d articles with empty clean_text", dropped)

    # --- Add relevance column ---
    df["relevance"] = "RELEVANT"

    # --- Reorder columns ---
    column_order = [
        "headline", "clean_text", "event_type", "stance",
        "impact_score", "materiality", "market_linkage",
        "time_sensitivity", "credibility",
        "source", "timestamp", "url", "region", "relevance",
        "label_source",
    ]
    df = df[[c for c in column_order if c in df.columns]]

    # --- Save ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    logger.info("Saved %d articles to: %s", len(df), OUTPUT_PATH)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXPANDED TRAINING DATA EXPORT — Summary")
    print("=" * 60)
    print(f"  Gold articles  : {gold_count}")
    print(f"  Silver articles: {silver_count}")
    print(f"  Total exported : {len(df)}")

    print("\n  Event type distribution:")
    for et, count in df["event_type"].value_counts().items():
        print(f"    {et:40s} {count}")

    print("\n  Stance distribution (3-class):")
    for st, count in df["stance"].value_counts().items():
        print(f"    {st:15s} {count}")

    print(f"\n  Impact score range: {df['impact_score'].min():.2f} – {df['impact_score'].max():.2f}")
    print(f"  Impact score mean : {df['impact_score'].mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

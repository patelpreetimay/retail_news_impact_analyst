"""
export_relevance_dataset.py — Export the Relevance Training Dataset
======================================================================

Pulls **every analysed article** from `data/rnia.db` (both relevance=0 and
relevance=1) and writes a single training CSV used by the binary
relevance classifier.

Why this exists
---------------
The gold export (`scripts/export_gold_dataset.py`) only emits the 1,817
relevance=1 rows — that's the gold dataset for the event/stance/impact
classifiers. The relevance model needs both classes, so we materialise a
separate CSV here.

Source rows
-----------
We take every row from `v2_analyses` where `processing_status` is one of:

    'done'      → the LLM saw it and scored relevance ∈ {0, 1}
    'filtered'  → the deterministic Stage-1 keyword filter dropped it
                  (junk_stub / body_too_short / no_keyword_match);
                  these are unambiguously relevance = 0

Rows with status 'pending' / 'needs_escalation' / 'failed' are SKIPPED —
they don't have a trustworthy relevance label.

Output
------
data/labeled_dataset/relevance_dataset.csv

Columns
-------
    article_id, headline, clean_text, source, region, timestamp, url,
    relevance        # int, {0, 1}
    relevance_source # 'gemini' | 'keyword_filter'  (provenance flag)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "rnia.db")
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "relevance_dataset.csv"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SQL = """
    SELECT
        a.article_id,
        a.headline,
        a.clean_text,
        a.source,
        a.region,
        a.published_at  AS timestamp,
        a.url,
        v.processing_status,
        v.relevance,
        v.filter_reason
    FROM articles a
    JOIN v2_analyses v ON v.article_id = a.article_id
    WHERE v.processing_status IN ('done', 'filtered')
    ORDER BY a.article_id
"""


def main() -> None:
    logger.info("=" * 60)
    logger.info("Export Relevance Training Dataset")
    logger.info("=" * 60)

    if not os.path.isfile(DB_PATH):
        logger.error("Database not found: %s", DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(SQL, conn)
    conn.close()
    logger.info("Pulled %d rows from v2_analyses", len(df))

    # Split by provenance and resolve the binary relevance label.
    done_mask = df["processing_status"] == "done"
    filtered_mask = df["processing_status"] == "filtered"

    # 'done' rows: trust v.relevance directly (Gemini-judged).
    done_df = df[done_mask].copy()
    # Drop rows whose relevance is NULL (shouldn't happen, but be safe).
    bad_done = done_df["relevance"].isna().sum()
    if bad_done:
        logger.warning("Dropping %d 'done' rows with NULL relevance.", bad_done)
        done_df = done_df.dropna(subset=["relevance"])
    done_df["relevance"] = done_df["relevance"].astype(int)
    done_df["relevance_source"] = "gemini"

    # 'filtered' rows: stage-1 keyword filter dropped them ⇒ relevance = 0.
    filtered_df = df[filtered_mask].copy()
    filtered_df["relevance"] = 0
    filtered_df["relevance_source"] = "keyword_filter"

    out = pd.concat([done_df, filtered_df], ignore_index=True)

    # Drop rows with empty headline+clean_text — nothing to learn from.
    before = len(out)
    out["clean_text"] = out["clean_text"].fillna("")
    out["headline"] = out["headline"].fillna("")
    out = out[(out["clean_text"].str.strip() != "") | (out["headline"].str.strip() != "")]
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %d rows with no text content.", dropped)

    # Final column order (drop the staging columns).
    cols = [
        "article_id", "headline", "clean_text",
        "source", "region", "timestamp", "url",
        "relevance", "relevance_source",
    ]
    out = out[cols].sort_values("article_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("RELEVANCE DATASET — Summary")
    print("=" * 60)
    print(f"  Total rows exported : {len(out)}")
    print(f"  Output path         : {OUTPUT_PATH}")

    print("\n  Relevance distribution:")
    for v, c in out["relevance"].value_counts().sort_index().items():
        pct = 100 * c / max(len(out), 1)
        print(f"    {v}: {c:>5d}  ({pct:5.1f}%)")

    print("\n  Provenance breakdown:")
    for src, c in out["relevance_source"].value_counts().items():
        print(f"    {src:18s} {c:>5d}")

    print("\n  Region breakdown:")
    for r, c in out["region"].value_counts().items():
        print(f"    {r:8s} {c:>5d}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
pipeline/ingest.py — Ingest scraped + cleaned articles into data/rnia.db
=========================================================================

Reads the cleaned CSV produced by ``preprocessing/clean_text.py``
(``data/processed_news/news_clean_dataset.csv``) and inserts new
articles into the ``articles`` table.

Idempotent: ``url`` has a UNIQUE constraint, so re-running this on the
same CSV is a no-op for already-ingested URLs.

Per row, the script computes / fills:
    - clean_text_hash  (sha256 of clean_text)
    - body_length      (len(clean_text))
    - is_junk_stub     (1 if body_length < 600, mirroring V2 convention)
    - published_at     (best-effort ISO-8601 normalisation of timestamp)
    - timestamp_is_known (1 if parsing succeeded)
    - scraped_at       (now, UTC)
    - source_pipeline_version  ('v2' — schema CHECK forbids other values)

If a source name is encountered that is not yet in the ``sources``
table, the row is auto-inserted with a default credibility of 0.75 and
a region inferred from the scraper's INDIAN_SOURCES set.

Returns the list of newly-inserted ``article_id`` integers so downstream
stages can process exactly the new arrivals.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
from dateutil import parser as dateparser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scrapers.news_scraper import INDIAN_SOURCES

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "rnia.db")
CLEAN_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv"
)

# Schema convention: bodies shorter than this are scraper failures
JUNK_STUB_THRESHOLD = 600
DEFAULT_NEW_SOURCE_CREDIBILITY = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _try_parse_ts(raw: str | None) -> tuple[str | None, int]:
    """Return (iso_utc, timestamp_is_known) — falls back to (None, 0)."""
    if not raw or pd.isna(raw):
        return None, 0
    try:
        dt = dateparser.parse(str(raw))
        if dt is None:
            return None, 0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds"), 1
    except (ValueError, TypeError, OverflowError):
        return None, 0


def _ensure_source(conn: sqlite3.Connection, name: str) -> None:
    """Insert a row into `sources` if the name isn't there yet."""
    if not name:
        return
    exists = conn.execute(
        "SELECT 1 FROM sources WHERE source_name = ?", (name,)
    ).fetchone()
    if exists:
        return
    region = "India" if name in INDIAN_SOURCES else "Global"
    conn.execute(
        "INSERT INTO sources (source_name, credibility, region_default, notes) "
        "VALUES (?, ?, ?, ?)",
        (name, DEFAULT_NEW_SOURCE_CREDIBILITY, region,
         "auto-seeded by pipeline/ingest.py"),
    )
    logger.info("auto-seeded source: %s (region=%s)", name, region)


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

def ingest_clean_csv(
    csv_path: str = CLEAN_CSV,
    db_path: str = DB_PATH,
) -> list[int]:
    """
    Ingest a cleaned-news CSV into the articles table.

    Parameters
    ----------
    csv_path : str
        Path to news_clean_dataset.csv (must contain at least:
        headline, clean_text, source, timestamp, url, region; article_text is optional).
    db_path : str
        Path to data/rnia.db.

    Returns
    -------
    list[int]
        article_id of every row that was newly inserted by this call.
        Already-present URLs are silently skipped.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Cleaned CSV not found: {csv_path}")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"headline", "clean_text", "source", "url", "region"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cleaned CSV missing columns: {missing}")

    if "article_text" not in df.columns:
        df["article_text"] = df["clean_text"]
    if "timestamp" not in df.columns:
        df["timestamp"] = None

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    n_attempted = 0
    n_inserted = 0
    n_skipped_existing = 0
    n_skipped_invalid = 0
    new_ids: list[int] = []

    for _, row in df.iterrows():
        url = str(row.get("url") or "").strip()
        headline = str(row.get("headline") or "").strip()
        clean_text = str(row.get("clean_text") or "").strip()
        source = str(row.get("source") or "").strip()
        region = str(row.get("region") or "").strip()
        if region not in ("India", "Global"):
            region = "India" if source in INDIAN_SOURCES else "Global"

        if not url or not headline or not clean_text or not source:
            n_skipped_invalid += 1
            continue
        n_attempted += 1

        # Already present?
        existing = cur.execute(
            "SELECT article_id FROM articles WHERE url = ?", (url,)
        ).fetchone()
        if existing:
            n_skipped_existing += 1
            continue

        _ensure_source(conn, source)

        article_text = str(row.get("article_text") or clean_text)
        body_length = len(clean_text)
        clean_hash = _sha256(clean_text)
        is_junk_stub = 1 if body_length < JUNK_STUB_THRESHOLD else 0
        ts_iso, ts_known = _try_parse_ts(row.get("timestamp"))

        try:
            cur.execute(
                """
                INSERT INTO articles (
                    url, headline, article_text, clean_text,
                    source, region,
                    timestamp_raw, published_at,
                    scraped_at, source_pipeline_version,
                    body_length, clean_text_hash,
                    is_junk_stub, timestamp_is_known
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url, headline, article_text, clean_text,
                    source, region,
                    str(row.get("timestamp") or "") or None,
                    ts_iso,
                    now_iso, "v2",          # schema CHECK constrains this column
                    body_length, clean_hash,
                    is_junk_stub, ts_known,
                ),
            )
            new_ids.append(int(cur.lastrowid))
            n_inserted += 1
        except sqlite3.IntegrityError as exc:
            # Race / unique violation under concurrent runs — treat as skip.
            logger.debug("skip url=%s: %s", url, exc)
            n_skipped_existing += 1

    conn.commit()
    conn.close()

    logger.info(
        "ingest summary: attempted=%d  inserted=%d  skipped_existing=%d  skipped_invalid=%d",
        n_attempted, n_inserted, n_skipped_existing, n_skipped_invalid,
    )
    return new_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ids = ingest_clean_csv()
    print()
    print("=" * 60)
    print(f"INGEST — {len(ids)} new article_ids ingested")
    print("=" * 60)
    if ids:
        print(f"  range: {min(ids)} – {max(ids)}")
        print(f"  first 5: {ids[:5]}")

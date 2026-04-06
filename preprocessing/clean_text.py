"""
clean_text.py — Text Preprocessing for RNIA
============================================

Reads the raw news dataset produced by ``scrapers/news_scraper.py``,
applies text-cleaning transformations, and saves the processed dataset.

Cleaning Steps:
    1. Remove residual HTML tags.
    2. Lowercase all text.
    3. Collapse extra whitespace (spaces, tabs, newlines) into single spaces.

Input:
    data/raw_news/news_raw_dataset.csv

Output:
    data/processed_news/news_clean_dataset.csv

Columns:
    headline, article_text, clean_text, source, timestamp, url, region
"""

import os
import sys
import re
import logging

import pandas as pd
from bs4 import BeautifulSoup

# Ensure project root on sys.path for package imports
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from preprocessing.relevance_filter import is_relevant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "data", "raw_news", "news_raw_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed_news")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "news_clean_dataset.csv")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cleaning Functions
# ---------------------------------------------------------------------------


def clean_html(text: str) -> str:
    """
    Strip any residual HTML tags from *text*.

    Uses BeautifulSoup to parse the string and return plain text.

    Parameters
    ----------
    text : str
        Raw text that may contain HTML markup.

    Returns
    -------
    str
        Text with all HTML tags removed.
    """
    if not isinstance(text, str) or not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text()


def normalize_text(text: str) -> str:
    """
    Normalize *text* by lowercasing and collapsing whitespace.

    Steps:
        1. Convert to lowercase.
        2. Replace sequences of whitespace (spaces, tabs, newlines) with a
           single space.
        3. Strip leading/trailing whitespace.

    Parameters
    ----------
    text : str
        Input text string.

    Returns
    -------
    str
        Cleaned and normalized text.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_pipeline(text: str) -> str:
    """
    Full cleaning pipeline: HTML removal → normalization.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    str
        Cleaned text ready for downstream NLP tasks.
    """
    text = clean_html(text)
    text = normalize_text(text)
    return text


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main():
    """Read the raw dataset, clean the text, and save the processed dataset."""
    logger.info("=" * 60)
    logger.info("RNIA — Text Preprocessing Started")
    logger.info("=" * 60)

    # --- Load raw dataset ---------------------------------------------------
    if not os.path.isfile(INPUT_FILE):
        logger.error("Raw dataset not found at: %s", INPUT_FILE)
        logger.error("Please run scrapers/news_scraper.py first.")
        return

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    logger.info("Loaded %d articles from: %s", len(df), INPUT_FILE)

    # --- Apply cleaning -----------------------------------------------------
    logger.info("Cleaning article text...")
    df["clean_text"] = df["article_text"].apply(clean_pipeline)

    # Drop rows where cleaned text is empty
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d articles with empty cleaned text.", dropped)

    # --- RNIA Step 1 — Relevance Filter -------------------------------------
    # Drop sports / entertainment / non-financial articles before they reach
    # the event classifier. Keeps only articles with real economic / market
    # impact, per the RNIA system spec.
    logger.info("Applying RNIA relevance filter...")
    before_rel = len(df)
    relevance = df.apply(
        lambda row: is_relevant(str(row.get("headline", "")), str(row.get("clean_text", ""))),
        axis=1,
    )
    df["relevance"] = relevance.apply(lambda t: "RELEVANT" if t[0] else "IRRELEVANT")
    df["relevance_reason"] = relevance.apply(lambda t: t[1])

    dropped_irrelevant = (df["relevance"] == "IRRELEVANT").sum()
    if dropped_irrelevant:
        # Log a few examples for transparency
        sample = df.loc[df["relevance"] == "IRRELEVANT", ["headline", "relevance_reason"]].head(5)
        for _, r in sample.iterrows():
            logger.info("  drop: %s — %s", str(r["headline"])[:90], r["relevance_reason"])

    df = df[df["relevance"] == "RELEVANT"].reset_index(drop=True)
    logger.info(
        "Relevance filter: kept %d / %d articles (dropped %d).",
        len(df), before_rel, dropped_irrelevant,
    )

    # --- Reorder columns -----------------------------------------------------
    column_order = [
        "headline", "article_text", "clean_text",
        "source", "timestamp", "url", "region",
        "relevance",  # carried through so backend can filter defensively
    ]
    # Keep only columns that exist (backward compat for older datasets)
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    # --- Save ----------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    logger.info("Processed dataset saved to: %s", OUTPUT_FILE)
    logger.info("Total cleaned articles: %d", len(df))

    # --- Quick stats ---------------------------------------------------------
    avg_len = df["clean_text"].str.len().mean()
    logger.info("Average clean_text length: %.0f characters", avg_len)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

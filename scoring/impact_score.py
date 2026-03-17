"""
impact_score.py — Impact Scoring Engine for RNIA
=================================================

Calculates an overall impact score for each financial news event based on
three equally-weighted components:

    impact_score = 0.4 × credibility + 0.3 × recency + 0.3 × materiality

Components:
    1. **Credibility**  — Source reliability (Reuters > CNBC > Yahoo Finance)
    2. **Recency**       — Time decay (more recent articles score higher)
    3. **Materiality**   — Event-type importance (Earnings > Product Announcement)

Score Range: 0.0 – 1.0

Usage:
    # Single article
    >>> from scoring.impact_score import calculate_impact_score
    >>> result = calculate_impact_score("Reuters", "2026-03-13T10:00:00", "earnings")
    >>> print(result)
    {'impact_score': 0.92, 'credibility': 0.95, 'recency': 1.0, 'materiality': 0.95}

    # Batch processing
    >>> from scoring.impact_score import process_dataset
    >>> process_dataset()
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


# ===========================================================================
# WEIGHT CONFIGURATION
# ===========================================================================

# Formula weights (must sum to 1.0)
WEIGHT_CREDIBILITY = 0.4
WEIGHT_RECENCY = 0.3
WEIGHT_MATERIALITY = 0.3


# ===========================================================================
# STEP 2 — CREDIBILITY SCORE
# ===========================================================================

# Source reliability scores (0.0 – 1.0)
SOURCE_CREDIBILITY = {
    "reuters":               0.95,
    "wall street journal":   0.94,
    "wsj":                   0.94,
    "bloomberg":             0.93,
    "financial times":       0.92,
    "cnbc":                  0.90,
    "marketwatch":           0.88,
    "yahoo finance":         0.85,
    "investing.com":         0.80,
}

# Default credibility for unknown sources
DEFAULT_CREDIBILITY = 0.70


def calculate_credibility(source: str) -> float:
    """
    Calculate the credibility score based on the news source.

    More established and reputable sources receive higher scores.

    Parameters
    ----------
    source : str
        Name of the news source (e.g. ``"Reuters"``, ``"CNBC"``).

    Returns
    -------
    float
        Credibility score between 0.0 and 1.0.

    Examples
    --------
    >>> calculate_credibility("Reuters")
    0.95
    >>> calculate_credibility("CNBC")
    0.90
    >>> calculate_credibility("Unknown Blog")
    0.70
    """
    if not source or not isinstance(source, str):
        return DEFAULT_CREDIBILITY

    # Normalise to lowercase for matching
    source_lower = source.strip().lower()

    # Check against known sources
    for known_source, score in SOURCE_CREDIBILITY.items():
        if known_source in source_lower:
            return score

    return DEFAULT_CREDIBILITY


# ===========================================================================
# STEP 3 — RECENCY SCORE
# ===========================================================================

def calculate_recency(timestamp: str) -> float:
    """
    Calculate the recency score based on the article's publication timestamp.

    More recent articles receive higher scores, reflecting their greater
    relevance to current market conditions.

    Scoring tiers:
        - Published today (< 24h)   → 1.0
        - 1–2 days old               → 0.8
        - 3–7 days old               → 0.6
        - Older than 7 days          → 0.4

    Parameters
    ----------
    timestamp : str
        Publication timestamp in ISO-like format
        (e.g. ``"2026-03-13T10:00:00"`` or ``"2026-03-13 10:00:00"``).

    Returns
    -------
    float
        Recency score between 0.4 and 1.0.

    Examples
    --------
    >>> calculate_recency("2026-03-13T08:00:00")  # today
    1.0
    >>> calculate_recency("2026-03-01T08:00:00")  # old
    0.4
    """
    if not timestamp or not isinstance(timestamp, str):
        return 0.4  # Default for missing timestamps

    # Parse the timestamp string
    now = datetime.now()
    parsed = None

    # Try multiple common date formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(timestamp.strip(), fmt)
            # Strip timezone info for comparison
            parsed = parsed.replace(tzinfo=None)
            break
        except ValueError:
            continue

    if parsed is None:
        logger.warning("Could not parse timestamp: %s — using default recency.", timestamp)
        return 0.4

    # Calculate age in days
    assert parsed is not None  # guaranteed by the None check above
    age = now - parsed
    days_old = age.total_seconds() / 86400  # Convert to fractional days

    # Assign recency tier
    if days_old < 1:
        return 1.0     # Published today (within 24 hours)
    elif days_old <= 2:
        return 0.8     # 1–2 days old
    elif days_old <= 7:
        return 0.6     # 3–7 days old
    else:
        return 0.4     # Older than 7 days


# ===========================================================================
# STEP 4 — MATERIALITY SCORE
# ===========================================================================

# Event-type importance weights (0.0 – 1.0)
EVENT_MATERIALITY = {
    "earnings":              0.95,
    "mergers_acquisitions":  0.90,
    "regulatory_action":     0.85,
    "leadership_change":     0.75,
    "legal_action":          0.70,
    "product_announcement":  0.65,
    "market_movement":       0.60,
}

# Default materiality for unknown event types
DEFAULT_MATERIALITY = 0.50


def calculate_materiality(event_type: str) -> float:
    """
    Calculate the materiality score based on the event type.

    Events with higher financial significance (e.g. earnings, M&A)
    receive higher scores.

    Parameters
    ----------
    event_type : str
        Financial event category (e.g. ``"earnings"``,
        ``"mergers_acquisitions"``).

    Returns
    -------
    float
        Materiality score between 0.0 and 1.0.

    Examples
    --------
    >>> calculate_materiality("earnings")
    0.95
    >>> calculate_materiality("product_announcement")
    0.65
    """
    if not event_type or not isinstance(event_type, str):
        return DEFAULT_MATERIALITY

    return EVENT_MATERIALITY.get(event_type.strip().lower(), DEFAULT_MATERIALITY)


# ===========================================================================
# STEP 5 — FINAL IMPACT SCORE
# ===========================================================================

def calculate_impact_score(
    source: str,
    timestamp: str,
    event_type: str,
) -> dict:
    """
    Calculate the final weighted impact score for a news article.

    Formula:
        ``impact_score = 0.4 × credibility + 0.3 × recency + 0.3 × materiality``

    Parameters
    ----------
    source : str
        News source name (e.g. ``"Reuters"``).
    timestamp : str
        Publication timestamp (e.g. ``"2026-03-13T10:00:00"``).
    event_type : str
        Financial event category (e.g. ``"earnings"``).

    Returns
    -------
    dict
        Dictionary with keys:
            - ``impact_score`` : float — Final weighted score (0.0 – 1.0)
            - ``credibility``  : float — Source credibility component
            - ``recency``      : float — Recency component
            - ``materiality``  : float — Event materiality component

    Examples
    --------
    >>> result = calculate_impact_score("Reuters", "2026-03-13T10:00:00", "earnings")
    >>> print(f"Impact: {result['impact_score']:.2f}")
    Impact: 0.97
    """
    # Calculate each component
    credibility = calculate_credibility(source)
    recency = calculate_recency(timestamp)
    materiality = calculate_materiality(event_type)

    # Weighted combination
    impact_score = (
        WEIGHT_CREDIBILITY * credibility
        + WEIGHT_RECENCY * recency
        + WEIGHT_MATERIALITY * materiality
    )

    # Clamp to [0, 1] (should already be in range, but just in case)
    impact_score = float(max(0.0, min(1.0, impact_score)))

    return {
        "impact_score": round(impact_score, 4),
        "credibility": round(credibility, 4),
        "recency": round(recency, 4),
        "materiality": round(materiality, 4),
    }


# ===========================================================================
# STEP 6 — BATCH PROCESSING
# ===========================================================================

# Default paths
INPUT_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv"
)
OUTPUT_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed_news", "news_with_impact_scores.csv"
)


def process_dataset(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Process an entire dataset and compute impact scores for all articles.

    Reads the input CSV, calculates impact scores for each row using the
    ``source``, ``timestamp``, and ``event_type`` columns, appends four
    new columns (``impact_score``, ``credibility``, ``recency``,
    ``materiality``), and saves the result.

    If the dataset does not have an ``event_type`` column, the function
    will use ``"market_movement"`` as a default for all articles.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file (default: processed clean dataset).
    output_csv : str
        Path to save the scored output CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with the original columns plus the four scoring columns.
    """
    logger.info("=" * 60)
    logger.info("RNIA — Impact Scoring Engine")
    logger.info("=" * 60)

    # Load dataset
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    logger.info("Loaded %d articles from: %s", len(df), input_csv)

    # Check for required columns
    has_event_type = "event_type" in df.columns
    if not has_event_type:
        logger.warning(
            "Column 'event_type' not found — using 'market_movement' "
            "as default for all articles."
        )

    # Calculate impact scores for each article
    impact_scores = []
    credibility_scores = []
    recency_scores = []
    materiality_scores = []

    for idx, row in df.iterrows():
        source = str(row.get("source", ""))
        timestamp = str(row.get("timestamp", ""))
        event_type = str(row.get("event_type", "market_movement")) if has_event_type else "market_movement"

        result = calculate_impact_score(source, timestamp, event_type)

        impact_scores.append(result["impact_score"])
        credibility_scores.append(result["credibility"])
        recency_scores.append(result["recency"])
        materiality_scores.append(result["materiality"])

    # Append new columns
    df["credibility"] = credibility_scores
    df["recency"] = recency_scores
    df["materiality"] = materiality_scores
    df["impact_score"] = impact_scores

    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved scored dataset to: %s", output_csv)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Impact Scoring — Summary")
    print("=" * 60)
    print(f"Articles processed : {len(df)}")
    print(f"Impact score range : {df['impact_score'].min():.4f} – {df['impact_score'].max():.4f}")
    print(f"Mean impact score  : {df['impact_score'].mean():.4f}")
    print(f"\nCredibility  — mean: {df['credibility'].mean():.4f}")
    print(f"Recency      — mean: {df['recency'].mean():.4f}")
    print(f"Materiality  — mean: {df['materiality'].mean():.4f}")

    # Distribution by source
    print("\nImpact scores by source:")
    print(df.groupby("source")["impact_score"].mean().to_string())

    print(f"\nOutput saved to: {output_csv}")
    print("=" * 60)

    return df


# ===========================================================================
# Main — Run batch processing
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Single article test
    print("=" * 60)
    print("Single Article Test")
    print("=" * 60)

    test_result = calculate_impact_score(
        source="Reuters",
        timestamp="2026-03-13T10:00:00",
        event_type="earnings",
    )
    print(f"  Source    : Reuters")
    print(f"  Timestamp : 2026-03-13T10:00:00")
    print(f"  Event     : earnings")
    print(f"  ---")
    print(f"  Credibility  : {test_result['credibility']}")
    print(f"  Recency      : {test_result['recency']}")
    print(f"  Materiality  : {test_result['materiality']}")
    print(f"  Impact Score : {test_result['impact_score']}")

    # Batch processing
    print()
    process_dataset()

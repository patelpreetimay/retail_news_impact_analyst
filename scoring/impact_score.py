"""
impact_score.py — Impact Scoring Engine for RNIA
=================================================

Calculates an overall impact score for each financial news event based on
stable importance plus smooth recency decay:

    base_importance = 0.6 * materiality + 0.4 * credibility
    impact_score = base_importance * (0.5 + 0.5 * recency)

Components:
    1. **Credibility**  — Source reliability (Reuters > CNBC > Yahoo Finance)
    2. **Recency**       — Smooth time decay (more recent articles score higher)
    3. **Materiality**   — Event-type importance (Earnings > Product Announcement)

Score Range: 0.0 – 1.0

Usage:
    # Single article
    >>> from scoring.impact_score import calculate_impact_score
    >>> result = calculate_impact_score("Reuters", "2026-03-13T10:00:00", "earnings")
    >>> sorted(result)
    ['credibility', 'impact_score', 'materiality', 'recency']

    # Batch processing
    >>> from scoring.impact_score import process_dataset
    >>> process_dataset()
"""

import os
import sys
import logging
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

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

# Base importance weights. Recency is applied as a multiplier so stale news
# visibly fades instead of only losing a small additive component.
WEIGHT_MATERIALITY = 0.6
WEIGHT_CREDIBILITY = 0.4

# Smooth recency decay:
#   recency = floor + (1 - floor) * 2 ** (-age_hours / half_life_hours)
# Approximate curve: 1 hour=0.99, 1 day=0.72, 1 week=0.13, 1 month=0.05.
RECENCY_HALF_LIFE_HOURS = 48.0
RECENCY_FLOOR = 0.05
DEFAULT_RECENCY = 0.5
RECENCY_MULTIPLIER_FLOOR = 0.5


# ===========================================================================
# STEP 2 — CREDIBILITY SCORE
# ===========================================================================

# Source reliability scores (0.0 – 1.0)
SOURCE_CREDIBILITY = {
    # Global sources
    "reuters":               0.95,
    "wall street journal":   0.94,
    "wsj":                   0.94,
    "bloomberg":             0.93,
    "financial times":       0.92,
    "cnbc":                  0.90,
    "marketwatch":           0.88,
    "yahoo finance":         0.85,
    "investing.com":         0.80,
    # Indian sources
    "economic times":        0.90,
    "moneycontrol":          0.88,
    "livemint":              0.88,
    "hindu businessline":    0.88,
    "ndtv profit":           0.86,
    "cnbctv18":              0.86,
    "business today":        0.84,
    "zee business":          0.82,
    "investing india":       0.80,
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

def _parse_timestamp(timestamp: str | datetime) -> datetime | None:
    """Parse common ISO/RSS timestamp formats into a datetime object."""
    if isinstance(timestamp, datetime):
        return timestamp

    if (
        not timestamp
        or not isinstance(timestamp, str)
        or timestamp.strip().lower() in ("nan", "nat", "none", "")
    ):
        return None

    raw = timestamp.strip()
    iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        pass

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
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def calculate_recency(timestamp: str | datetime, now: datetime | None = None) -> float:
    """
    Calculate a smooth recency score from the publication timestamp.

    The score follows a 48-hour half-life and bottoms out at 0.05:
    1 hour ~= 0.99, 1 day ~= 0.72, 1 week ~= 0.13, 1 month ~= 0.05.
    Missing or unparseable timestamps return a neutral default of 0.5.
    """
    if (
        not timestamp
        or (
            isinstance(timestamp, str)
            and timestamp.strip().lower() in ("nan", "nat", "none", "")
        )
    ):
        return DEFAULT_RECENCY

    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        logger.warning("Could not parse timestamp: %s â€” using default recency.", timestamp)
        return DEFAULT_RECENCY

    if parsed.tzinfo is not None:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        age = current.astimezone(timezone.utc) - parsed.astimezone(timezone.utc)
    else:
        current = now or datetime.now()
        if current.tzinfo is not None:
            current = current.astimezone().replace(tzinfo=None)
        age = current - parsed

    age_hours = max(0.0, age.total_seconds() / 3600.0)
    recency = RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * math.pow(
        0.5,
        age_hours / RECENCY_HALF_LIFE_HOURS,
    )
    return round(float(max(0.0, min(1.0, recency))), 4)


# ===========================================================================
# STEP 4 — MATERIALITY SCORE
# ===========================================================================

# Event-type importance weights (0.0 – 1.0)
EVENT_MATERIALITY = {
    "earnings":                          0.95,
    "mergers_acquisitions":              0.90,
    "macroeconomic_geopolitical":        0.90,
    "regulatory_action":                 0.85,
    "market_sentiment_investor_action":  0.80,
    "leadership_change":                 0.75,
    "legal_action":                      0.70,
    "product_announcement":              0.65,
    "other":                             0.55,
    # Legacy V1 names (backward compat)
    "market_movement":                   0.60,
    "unclassified":                      0.30,
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

def calculate_impact_from_components(
    materiality: float,
    credibility: float,
    recency: float,
) -> float:
    """
    Compose final impact from stable importance and live recency.

    Formula:
        impact = (0.6 * materiality + 0.4 * credibility)
                 * (0.5 + 0.5 * recency)
    """
    materiality = float(max(0.0, min(1.0, materiality)))
    credibility = float(max(0.0, min(1.0, credibility)))
    recency = float(max(0.0, min(1.0, recency)))

    base_importance = (
        WEIGHT_MATERIALITY * materiality
        + WEIGHT_CREDIBILITY * credibility
    )
    freshness_multiplier = (
        RECENCY_MULTIPLIER_FLOOR
        + (1.0 - RECENCY_MULTIPLIER_FLOOR) * recency
    )
    impact_score = base_importance * freshness_multiplier
    return round(float(max(0.0, min(1.0, impact_score))), 4)


def calculate_impact_score(
    source: str,
    timestamp: str,
    event_type: str,
) -> dict:
    """
    Calculate the final weighted impact score for a news article.

    Formula:
        ``impact_score = (0.6 × materiality + 0.4 × credibility)
        × (0.5 + 0.5 × recency)``

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
    >>> 0.0 <= result["impact_score"] <= 1.0
    True
    """
    # Calculate each component
    credibility = calculate_credibility(source)
    recency = calculate_recency(timestamp)
    materiality = calculate_materiality(event_type)

    impact_score = calculate_impact_from_components(
        materiality=materiality,
        credibility=credibility,
        recency=recency,
    )

    return {
        "impact_score": round(float(impact_score), 4),
        "credibility": round(float(credibility), 4),
        "recency": round(float(recency), 4),
        "materiality": round(float(materiality), 4),
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

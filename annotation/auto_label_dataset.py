"""
auto_label_dataset.py — Automatic Keyword-Based Labeling for RNIA
==================================================================

Reads the cleaned dataset and automatically assigns event_type and
stance labels to every article using keyword matching on the clean_text.

Input:
    data/processed_news/news_clean_dataset.csv

Output:
    data/labeled_dataset/financial_news_labeled.csv

Columns:
    headline, clean_text, event_type, stance, source, timestamp, url
"""

import os
import sys
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Paths
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "labeled_dataset")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "financial_news_labeled.csv")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# STEP 2 — Event Type Keywords
# ===========================================================================

# Each event type maps to a list of keywords.
# The category with the MOST keyword matches wins (scoring-based).
EVENT_KEYWORDS: dict[str, list[str]] = {
    "earnings": [
        "earnings", "revenue", "quarterly results", "profit", "guidance",
        "eps", "fiscal", "income", "dividend", "financial results",
        "net income", "operating income", "gross margin", "beat estimates",
        "missed estimates", "quarterly earnings", "annual report",
        "earnings per share", "revenue growth", "profit margin",
    ],
    "leadership_change": [
        "ceo", "cfo", "coo", "cto", "resignation", "appointed",
        "leadership", "executive", "chairman", "board of directors",
        "stepping down", "hire", "successor", "interim chief",
        "management shakeup", "new president", "chief executive",
        "executive departure", "board member", "named as",
    ],
    "regulatory_action": [
        "regulator", "sec", "investigation", "compliance", "regulation",
        "federal reserve", "antitrust", "sanction", "enforcement",
        "regulatory approval", "policy change", "fine", "penalty",
        "fda approval", "ftc", "regulatory filing", "central bank",
        "rate decision", "interest rate", "government intervention",
    ],
    "mergers_acquisitions": [
        "merger", "acquisition", "takeover", "buyout", "deal",
        "stake", "consolidation", "joint venture", "acquire",
        "bid", "offer to buy", "purchase agreement", "divestiture",
        "spin-off", "hostile takeover", "friendly merger",
        "all-stock deal", "cash deal", "combined entity",
    ],
    "legal_action": [
        "lawsuit", "court", "settlement", "litigation", "sue",
        "ruling", "verdict", "penalty", "fraud", "legal dispute",
        "class action", "indictment", "plea deal", "injunction",
        "patent infringement", "antitrust lawsuit", "regulatory fine",
        "criminal charges", "wrongful termination", "defamation",
    ],
    "product_announcement": [
        "launch", "release", "unveil", "product", "innovation",
        "patent", "prototype", "rollout", "new feature",
        "product launch", "new model", "next generation",
        "product line", "product reveal", "software update",
        "hardware release", "beta launch", "product roadmap",
    ],
    "market_movement": [
        "market outlook", "industry trend", "stock market", "sector trend",
        "trading", "index", "rally", "sell-off", "volatility",
        "bull market", "bear market", "wall street", "market cap",
        "downturn", "recovery", "correction", "all-time high",
        "market sentiment", "investor confidence", "economic indicator",
    ],
}

# Default event type if no keywords match at all
DEFAULT_EVENT = "market_movement"

# Priority order for tie-breaking (lower index = higher priority)
EVENT_PRIORITY = [
    "mergers_acquisitions",
    "leadership_change",
    "legal_action",
    "regulatory_action",
    "product_announcement",
    "earnings",
    "market_movement",
]


# ===========================================================================
# STEP 3 — Stance Keywords
# ===========================================================================

POSITIVE_KEYWORDS = [
    "growth", "strong", "record", "surge", "positive",
    "gain", "rise", "boost", "upbeat", "beat", "exceed",
    "optimistic", "upgrade", "outperform", "bullish",
]

NEGATIVE_KEYWORDS = [
    "loss", "decline", "drop", "weak", "lawsuit",
    "fall", "cut", "downgrade", "miss", "crash",
    "warning", "risk", "concern", "bearish", "slump",
]


# ===========================================================================
# Labeling Functions
# ===========================================================================

def detect_event_type(text: str) -> str:
    """
    Detect the financial event type using **scoring-based** keyword matching.

    Counts keyword hits for EVERY category and picks the one with the
    highest score.  If two or more categories are tied, the one appearing
    earlier in ``EVENT_PRIORITY`` wins.  If no keywords match at all, the
    function falls back to ``DEFAULT_EVENT`` (``"market_movement"``).

    Parameters
    ----------
    text : str
        Cleaned article text (lowercase).

    Returns
    -------
    str
        One of the 7 event categories, or ``"market_movement"`` as default.
    """
    if not text or not isinstance(text, str):
        return DEFAULT_EVENT

    text_lower = text.lower()

    # Count keyword matches for every category
    scores: dict[str, int] = {}
    for event_type, keywords in EVENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[event_type] = score

    max_score = max(scores.values())

    # No keywords matched → fall back to default
    if max_score == 0:
        return DEFAULT_EVENT

    # Collect all categories that achieved the max score
    top_categories = [cat for cat, s in scores.items() if s == max_score]

    # Tie-break using the priority list
    if len(top_categories) == 1:
        return top_categories[0]

    for cat in EVENT_PRIORITY:
        if cat in top_categories:
            return cat

    return top_categories[0]


def detect_stance(text: str) -> str:
    """
    Detect the sentiment stance from article text using keyword matching.

    Counts positive and negative keyword hits; the higher count wins.
    If tied or no matches, returns ``"neutral"``.

    Parameters
    ----------
    text : str
        Cleaned article text (lowercase).

    Returns
    -------
    str
        ``"positive"``, ``"negative"``, or ``"neutral"``.
    """
    if not text or not isinstance(text, str):
        return "neutral"

    text_lower = text.lower()

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


# ===========================================================================
# STEP 4 & 5 — Apply Labels and Save
# ===========================================================================

def auto_label_dataset(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Load the cleaned dataset, apply keyword-based labels, and save.

    Parameters
    ----------
    input_csv : str
        Path to the cleaned dataset CSV.
    output_csv : str
        Path to save the labeled dataset CSV.

    Returns
    -------
    pd.DataFrame
        Labeled DataFrame with columns:
        headline, clean_text, event_type, stance, source, timestamp, url.
    """
    logger.info("=" * 60)
    logger.info("RNIA — Auto-Labeling Dataset")
    logger.info("=" * 60)

    # Load
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    logger.info("Loaded %d articles from: %s", len(df), input_csv)

    # Apply labels
    logger.info("Detecting event types...")
    df["event_type"] = df["clean_text"].apply(detect_event_type)

    logger.info("Detecting stances...")
    df["stance"] = df["clean_text"].apply(detect_stance)

    # Keep only the required columns
    output_columns = ["headline", "clean_text", "event_type", "stance", "source", "timestamp", "url"]
    df = df[output_columns]

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("Labeled dataset saved to: %s", output_csv)

    # Summary
    print("\n" + "=" * 60)
    print("Auto-Labeling Summary")
    print("=" * 60)
    print(f"Total articles labeled: {len(df)}")

    print("\nEvent Type Distribution:")
    for event, count in df["event_type"].value_counts().items():
        print(f"  {event:25s} : {count}")

    print("\nStance Distribution:")
    for stance, count in df["stance"].value_counts().items():
        print(f"  {stance:25s} : {count}")

    print(f"\nOutput: {output_csv}")
    print("=" * 60)

    return df


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    auto_label_dataset()

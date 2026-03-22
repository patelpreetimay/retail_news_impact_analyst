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
    headline, clean_text, event_type, stance, source, timestamp, url, region
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
        # Core earnings terms
        "earnings", "revenue", "quarterly results", "profit", "guidance",
        "eps", "fiscal", "income", "dividend", "financial results",
        "net income", "operating income", "gross margin", "beat estimates",
        "missed estimates", "quarterly earnings", "annual report",
        "earnings per share", "revenue growth", "profit margin",
        # Extended earnings vocabulary
        "top line", "bottom line", "ebitda", "operating profit",
        "same-store sales", "comp sales", "sales growth", "revenue miss",
        "revenue beat", "profit warning", "earnings surprise",
        "earnings miss", "earnings beat", "fiscal year", "fiscal quarter",
        "q1 results", "q2 results", "q3 results", "q4 results",
        "annual results", "half-year results", "full-year results",
        "reported revenue", "reported earnings", "reported profit",
        "net profit", "gross profit", "operating margin",
        "profit after tax", "profit before tax", "return on equity",
        "earnings outlook", "revenue outlook", "sales forecast",
        "raised guidance", "lowered guidance", "reaffirmed guidance",
        "beat expectations", "missed expectations", "exceeded estimates",
        "fell short", "above consensus", "below consensus",
        "dividend payout", "dividend yield", "share buyback",
        "stock repurchase", "cash flow", "free cash flow",
    ],
    "leadership_change": [
        # Core leadership terms
        "ceo", "cfo", "coo", "cto", "resignation", "appointed",
        "leadership", "executive", "chairman", "board of directors",
        "stepping down", "hire", "successor", "interim chief",
        "management shakeup", "new president", "chief executive",
        "executive departure", "board member", "named as",
        # Extended leadership vocabulary
        "managing director", "vice president", "senior vice president",
        "chief operating officer", "chief financial officer",
        "chief technology officer", "chief marketing officer",
        "board reshuffle", "boardroom", "executive team",
        "c-suite", "top management", "new leadership",
        "leadership transition", "ceo transition", "ousted",
        "fired", "terminated", "replaced", "new ceo",
        "new cfo", "incoming ceo", "outgoing ceo",
        "co-founder", "founder steps down", "promoter",
        "independent director", "non-executive director",
        "management change", "new appointment", "key appointment",
    ],
    "regulatory_action": [
        # Core regulatory terms
        "regulator", "sec", "investigation", "compliance", "regulation",
        "federal reserve", "antitrust", "sanction", "enforcement",
        "regulatory approval", "policy change", "fine", "penalty",
        "fda approval", "ftc", "regulatory filing", "central bank",
        "rate decision", "interest rate", "government intervention",
        # Extended regulatory vocabulary
        "regulatory crackdown", "regulatory scrutiny", "deregulation",
        "new regulation", "proposed regulation", "rule change",
        "congressional hearing", "senate hearing", "legislation",
        "executive order", "trade policy", "tariff", "import duty",
        "export ban", "trade war", "trade restriction", "subsidy",
        "tax reform", "tax cut", "tax hike", "stimulus",
        "bailout", "quantitative easing", "rate hike", "rate cut",
        "inflation target", "price control", "price cap",
        "environmental regulation", "emissions standard",
        "data privacy", "gdpr", "antitrust probe", "monopoly",
        "market manipulation", "insider trading probe",
        # India-specific regulators
        "sebi", "rbi", "reserve bank of india", "nse", "bse",
        "irdai", "repo rate", "monetary policy", "fiscal policy",
        "gst", "union budget", "niti aayog", "crr", "slr",
        "reverse repo", "statutory liquidity", "cash reserve ratio",
        "sebi order", "rbi circular", "rbi policy",
        "trai", "cci", "competition commission",
        "ministry of finance", "finance ministry",
    ],
    "mergers_acquisitions": [
        # Core M&A terms
        "merger", "acquisition", "takeover", "buyout", "deal",
        "stake", "consolidation", "joint venture", "acquire",
        "bid", "offer to buy", "purchase agreement", "divestiture",
        "spin-off", "hostile takeover", "friendly merger",
        "all-stock deal", "cash deal", "combined entity",
        # Extended M&A vocabulary
        "leveraged buyout", "lbo", "management buyout", "mbo",
        "acquirer", "target company", "merger agreement",
        "definitive agreement", "letter of intent", "due diligence",
        "synergy", "synergies", "cost synergies", "revenue synergies",
        "strategic acquisition", "bolt-on acquisition", "tuck-in",
        "majority stake", "minority stake", "controlling stake",
        "share swap", "equity swap", "tender offer",
        "open offer", "delisting", "going private",
        "breakup fee", "termination fee", "anti-dilution",
        "strategic partnership", "strategic alliance",
        "asset sale", "asset purchase", "carve-out",
        "reverse merger", "special purpose acquisition", "spac",
        "merger of equals", "absorbed by", "merged with",
        "acquired by", "to acquire", "agreed to buy",
        "buying spree", "deal value", "enterprise value",
    ],
    "legal_action": [
        # Core legal terms
        "lawsuit", "court", "settlement", "litigation", "sue",
        "ruling", "verdict", "penalty", "fraud", "legal dispute",
        "class action", "indictment", "plea deal", "injunction",
        "patent infringement", "antitrust lawsuit", "regulatory fine",
        "criminal charges", "wrongful termination", "defamation",
        # Extended legal vocabulary
        "sued", "suing", "filed suit", "legal proceedings",
        "court order", "court ruling", "judge ruled", "jury",
        "appeal", "appellate", "supreme court", "high court",
        "tribunal", "arbitration", "mediation", "legal battle",
        "legal challenge", "legal claim", "damages",
        "compensatory damages", "punitive damages", "restitution",
        "consent decree", "cease and desist", "restraining order",
        "securities fraud", "accounting fraud", "wire fraud",
        "embezzlement", "bribery", "corruption", "money laundering",
        "whistleblower", "class action lawsuit", "shareholder lawsuit",
        "derivative action", "breach of contract", "copyright",
        "trademark", "intellectual property", "trade secret",
        "insider trading", "market manipulation",
        "guilty plea", "not guilty", "convicted", "acquitted",
        "sentenced", "probation", "prison",
    ],
    "product_announcement": [
        # Core product terms
        "launch", "release", "unveil", "product", "innovation",
        "patent", "prototype", "rollout", "new feature",
        "product launch", "new model", "next generation",
        "product line", "product reveal", "software update",
        "hardware release", "beta launch", "product roadmap",
        # Extended product vocabulary
        "launched", "unveiled", "introduced", "announced",
        "new product", "new service", "new platform",
        "new version", "upgrade", "redesign", "revamp",
        "flagship product", "flagship device", "flagship model",
        "research and development", "r&d", "clinical trial",
        "fda clearance", "drug approval", "pipeline drug",
        "phase 1", "phase 2", "phase 3", "clinical data",
        "breakthrough", "disruptive", "cutting-edge",
        "artificial intelligence", "machine learning",
        "electric vehicle", "ev launch", "autonomous",
        "5g", "cloud computing", "saas", "platform update",
        "app launch", "marketplace", "subscription service",
        "expansion", "new market", "entered the market",
        "technology partnership", "tech stack",
    ],
    "market_movement": [
        # Core market terms
        "market outlook", "industry trend", "stock market", "sector trend",
        "trading", "index", "rally", "sell-off", "volatility",
        "bull market", "bear market", "wall street", "market cap",
        "downturn", "recovery", "correction", "all-time high",
        "market sentiment", "investor confidence", "economic indicator",
        # Extended market vocabulary
        "stock price", "share price", "stock surged", "stock plunged",
        "stock jumped", "stock fell", "stock dropped", "stock rose",
        "stocks rally", "stocks tumble", "stocks soar", "stocks sink",
        "market rally", "market crash", "market correction",
        "trading volume", "market volatility", "vix",
        "futures", "options", "derivatives", "short selling",
        "margin call", "circuit breaker", "trading halt",
        "ipo", "initial public offering", "secondary offering",
        "follow-on offering", "block deal", "bulk deal",
        "52-week high", "52-week low", "new high", "new low",
        "market breadth", "advance-decline", "market turnover",
        "sector rotation", "flight to safety", "risk-on", "risk-off",
        "yield curve", "treasury", "bond market", "fixed income",
        "commodity", "crude oil", "gold price", "oil price",
        "forex", "currency", "dollar index", "rupee",
        "gdp", "unemployment", "inflation", "cpi", "pmi",
        "consumer confidence", "retail sales", "manufacturing",
        "recession", "stagflation", "soft landing", "hard landing",
        # India-specific market terms
        "sensex", "nifty", "nifty 50", "bank nifty", "dalal street",
        "fii", "dii", "midcap", "smallcap", "largecap",
        "nifty it", "nifty bank", "nifty pharma", "nifty auto",
        "opening bell", "closing bell", "pre-market", "after-hours",
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
    # Core positive
    "growth", "strong", "record", "surge", "positive",
    "gain", "rise", "boost", "upbeat", "beat", "exceed",
    "optimistic", "upgrade", "outperform", "bullish",
    # Extended positive
    "soar", "rally", "jumped", "surged", "climbed",
    "robust", "stellar", "impressive", "exceeded expectations",
    "beat estimates", "above consensus", "raised guidance",
    "upside", "breakout", "momentum", "recovery",
    "expanded", "improved", "higher", "increased",
    "profit growth", "revenue growth", "strong demand",
    "record high", "all-time high", "new high",
    "outperformed", "topped estimates", "better than expected",
    "accelerated", "thriving", "booming", "flourishing",
    "dividend hike", "raised dividend", "buyback",
    "upgrade", "buy rating", "overweight", "accumulate",
    "opportunity", "confidence", "encouraging", "promising",
]

NEGATIVE_KEYWORDS = [
    # Core negative
    "loss", "decline", "drop", "weak", "lawsuit",
    "fall", "cut", "downgrade", "miss", "crash",
    "warning", "risk", "concern", "bearish", "slump",
    # Extended negative
    "plunge", "tumble", "sank", "plummeted", "collapsed",
    "disappointing", "missed estimates", "below consensus",
    "lowered guidance", "profit warning", "revenue miss",
    "downside", "selloff", "sell-off", "correction",
    "contracted", "deteriorated", "lower", "decreased",
    "deficit", "debt", "bankruptcy", "default", "insolvency",
    "layoffs", "job cuts", "restructuring", "cost-cutting",
    "underperform", "sell rating", "underweight", "reduce",
    "fraud", "scandal", "investigation", "probe",
    "penalty", "fine", "violation", "breach",
    "recession", "downturn", "slowdown", "stagnation",
    "threat", "uncertainty", "headwind", "pressure",
    "negative outlook", "credit downgrade", "junk status",
    "worst", "lowest", "record low", "crisis",
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
    output_columns = [
        "headline", "clean_text", "event_type", "stance",
        "source", "timestamp", "url", "region", "relevance",
    ]
    # Keep only columns that exist (backward compat for older datasets)
    output_columns = [c for c in output_columns if c in df.columns]
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

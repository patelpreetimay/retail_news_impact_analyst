"""
explanation_generator.py — Explanation Generator for RNIA
=========================================================

Converts the technical outputs of the RNIA system (event classification,
stance detection, and impact scoring) into human-readable explanations
suitable for retail investors.

Inputs (per article):
    event_type, stance, impact_score, credibility, recency, materiality

Outputs:
    A structured dictionary containing the event type, stance, impact
    score, and a natural-language explanation paragraph.

Usage:
    # Single article
    >>> from reporting.explanation_generator import generate_explanation
    >>> result = generate_explanation(
    ...     event_type="earnings", stance="positive",
    ...     impact_score=0.82, credibility=0.95,
    ...     recency=1.0, materiality=0.95,
    ... )
    >>> print(result["explanation"])

    # Batch processing
    >>> from reporting.explanation_generator import generate_reports_for_dataset
    >>> generate_reports_for_dataset()
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

logger = logging.getLogger(__name__)


# ===========================================================================
# HELPER — Human-readable label mappings
# ===========================================================================

# Event type to human-friendly description
EVENT_DESCRIPTIONS = {
    "earnings":              "an earnings and financial results",
    "leadership_change":     "a leadership change or executive appointment",
    "regulatory_action":     "a regulatory action or government policy",
    "mergers_acquisitions":  "a merger, acquisition, or corporate takeover",
    "legal_action":          "a legal action, lawsuit, or litigation",
    "product_announcement":  "a product announcement or launch",
    "market_movement":       "a stock market movement or trading activity",
}

# Stance to sentiment descriptor
STANCE_DESCRIPTORS = {
    "positive": {
        "sentiment": "positive",
        "implication": "favorable sentiment toward the company or market",
        "tone": "optimistic",
    },
    "negative": {
        "sentiment": "negative",
        "implication": "unfavorable outlook or potential risk factors",
        "tone": "cautious",
    },
    "neutral": {
        "sentiment": "neutral",
        "implication": "a balanced or factual reporting without strong bias",
        "tone": "measured",
    },
}

# Credibility descriptors based on score thresholds
def _credibility_phrase(score: float) -> str:
    """Return a human-readable phrase describing the source credibility."""
    if score >= 0.90:
        return "a highly credible and well-established financial source"
    elif score >= 0.80:
        return "a credible financial news source"
    elif score >= 0.70:
        return "a moderately credible news source"
    else:
        return "a less established news source"


# Recency descriptors based on score thresholds
def _recency_phrase(score: float) -> str:
    """Return a human-readable phrase describing the news recency."""
    if score >= 1.0:
        return "This is very recent news published today"
    elif score >= 0.8:
        return "This news was published within the last couple of days"
    elif score >= 0.6:
        return "This news is from the past week"
    else:
        return "This news is more than a week old"


# Impact level descriptors
def _impact_phrase(score: float) -> str:
    """Return a human-readable phrase describing the impact level."""
    if score >= 0.80:
        return "strong potential impact on market sentiment and investment decisions"
    elif score >= 0.60:
        return "moderate potential impact on market conditions"
    elif score >= 0.40:
        return "limited but notable potential impact"
    else:
        return "relatively low potential market impact"


# Materiality descriptors
def _materiality_phrase(score: float) -> str:
    """Return a human-readable phrase describing the event's materiality."""
    if score >= 0.85:
        return "high material importance in financial markets"
    elif score >= 0.70:
        return "notable financial significance"
    elif score >= 0.60:
        return "moderate financial relevance"
    else:
        return "limited financial materiality"


# ===========================================================================
# STEP 3 — GENERATE EXPLANATION TEXT
# ===========================================================================

def generate_explanation(
    event_type: str,
    stance: str,
    impact_score: float,
    credibility: float,
    recency: float,
    materiality: float,
) -> dict:
    """
    Generate a structured, human-readable explanation for a single article.

    Converts the technical classification and scoring outputs into a
    natural-language paragraph that a retail investor can understand.

    Parameters
    ----------
    event_type : str
        Classified financial event category (e.g. ``"earnings"``).
    stance : str
        Detected sentiment stance (``"positive"``, ``"negative"``,
        or ``"neutral"``).
    impact_score : float
        Overall weighted impact score (0.0 – 1.0).
    credibility : float
        Source credibility component (0.0 – 1.0).
    recency : float
        Recency component (0.0 – 1.0).
    materiality : float
        Event materiality component (0.0 – 1.0).

    Returns
    -------
    dict
        Dictionary with keys:
            - ``event_type``   : str
            - ``stance``       : str
            - ``impact_score`` : str (formatted)
            - ``explanation``  : str (natural-language paragraph)

    Examples
    --------
    >>> result = generate_explanation(
    ...     "earnings", "positive", 0.82, 0.95, 1.0, 0.95
    ... )
    >>> print(result["explanation"])
    """
    # Normalise inputs
    event_type_lower = str(event_type).strip().lower()
    stance_lower = str(stance).strip().lower()
    impact = float(impact_score)
    cred = float(credibility)
    rec = float(recency)
    mat = float(materiality)

    # Look up human-friendly descriptors
    event_desc = EVENT_DESCRIPTIONS.get(event_type_lower, f"a {event_type_lower} event")
    stance_info = STANCE_DESCRIPTORS.get(stance_lower, STANCE_DESCRIPTORS["neutral"])

    # Build the explanation paragraph
    explanation_parts = [
        # Sentence 1 — Event type + source credibility
        f"This article discusses {event_desc} event "
        f"reported by {_credibility_phrase(cred)}.",

        # Sentence 2 — Stance interpretation
        f"The {stance_info['sentiment']} stance indicates "
        f"{stance_info['implication']}, "
        f"suggesting a {stance_info['tone']} tone in the reporting.",

        # Sentence 3 — Recency context
        f"{_recency_phrase(rec)}, which "
        + ("enhances its relevance to current market conditions."
           if rec >= 0.8
           else "may reduce its immediate relevance to current market conditions."),

        # Sentence 4 — Materiality
        f"The event carries {_materiality_phrase(mat)}.",

        # Sentence 5 — Overall impact assessment
        f"Overall, the event has {_impact_phrase(impact)} "
        f"(impact score: {impact:.2f}/1.00).",
    ]

    explanation = " ".join(explanation_parts)

    # Return structured result (Step 4)
    return {
        "event_type": event_type_lower,
        "stance": stance_lower,
        "impact_score": f"{impact:.2f}",
        "explanation": explanation,
    }


# ===========================================================================
# STEP 5 — BATCH REPORT GENERATION
# ===========================================================================

# Default paths
INPUT_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed_news", "news_with_impact_scores.csv"
)
OUTPUT_CSV = os.path.join(
    PROJECT_ROOT, "data", "final_outputs", "news_analysis_report.csv"
)


def generate_reports_for_dataset(
    input_csv: str = INPUT_CSV,
    output_csv: str = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Generate human-readable explanations for every article in a dataset.

    Reads the scored dataset, generates an explanation for each row,
    adds an ``explanation`` column, and saves the final report CSV.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV (must have columns: ``event_type``,
        ``stance``, ``impact_score``, ``credibility``, ``recency``,
        ``materiality``).
    output_csv : str
        Path to save the output report CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ``explanation`` column appended.

    Output Columns:
        headline, event_type, stance, impact_score, explanation
        (plus all original columns from the input)
    """
    logger.info("=" * 60)
    logger.info("RNIA — Explanation Generator")
    logger.info("=" * 60)

    # Load dataset
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(
            f"Scored dataset not found: {input_csv}\n"
            "Run scoring/impact_score.py first."
        )

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    logger.info("Loaded %d articles from: %s", len(df), input_csv)

    # Verify required columns
    required_cols = ["event_type", "stance", "impact_score", "credibility", "recency", "materiality"]
    missing = [c for c in required_cols if c not in df.columns]

    # If event_type or stance are missing, use defaults
    if "event_type" not in df.columns:
        logger.warning("Column 'event_type' missing — defaulting to 'market_movement'.")
        df["event_type"] = "market_movement"
    if "stance" not in df.columns:
        logger.warning("Column 'stance' missing — defaulting to 'neutral'.")
        df["stance"] = "neutral"
    if "credibility" not in df.columns or "recency" not in df.columns or "materiality" not in df.columns:
        logger.warning(
            "Score component columns missing. Run scoring/impact_score.py first "
            "to add credibility, recency, and materiality columns."
        )
        # Use defaults
        df["credibility"] = df.get("credibility", 0.70)
        df["recency"] = df.get("recency", 0.40)
        df["materiality"] = df.get("materiality", 0.50)
    if "impact_score" not in df.columns:
        df["impact_score"] = 0.50

    # Generate explanations for each article
    explanations = []
    for idx, row in df.iterrows():
        result = generate_explanation(
            event_type=str(row.get("event_type", "market_movement")),
            stance=str(row.get("stance", "neutral")),
            impact_score=float(row.get("impact_score", 0.5)),
            credibility=float(row.get("credibility", 0.7)),
            recency=float(row.get("recency", 0.4)),
            materiality=float(row.get("materiality", 0.5)),
        )
        explanations.append(result["explanation"])

    # Add explanation column
    df["explanation"] = explanations

    # Save output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("Saved report to: %s", output_csv)

    # Print summary
    print("\n" + "=" * 60)
    print("Explanation Generator — Summary")
    print("=" * 60)
    print(f"Articles processed : {len(df)}")
    print(f"Output saved to    : {output_csv}")

    # Show one example row (Step 6)
    print("\n" + "-" * 60)
    print("Example Output Row:")
    print("-" * 60)
    example = df.iloc[0]
    print(f"  Headline     : {example.get('headline', 'N/A')}")
    print(f"  Event Type   : {example.get('event_type', 'N/A')}")
    print(f"  Stance       : {example.get('stance', 'N/A')}")
    print(f"  Impact Score : {example.get('impact_score', 'N/A')}")
    print(f"  Explanation  : {example.get('explanation', 'N/A')}")
    print("=" * 60)

    return df


# ===========================================================================
# Main — Run batch report generation
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Single article test
    print("=" * 60)
    print("Single Explanation Test")
    print("=" * 60)

    result = generate_explanation(
        event_type="earnings",
        stance="positive",
        impact_score=0.82,
        credibility=0.95,
        recency=1.0,
        materiality=0.95,
    )
    print(f"\n  Event Type   : {result['event_type']}")
    print(f"  Stance       : {result['stance']}")
    print(f"  Impact Score : {result['impact_score']}")
    print(f"\n  Explanation:")
    print(f"  {result['explanation']}")

    # Batch processing
    print()
    generate_reports_for_dataset()

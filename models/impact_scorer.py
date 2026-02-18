"""
impact_scorer.py — Impact Scoring for RNIA
===========================================

Computes a composite impact score (0–10) for each news article based on:

    1. **Event severity weight** — how disruptive the event category typically is
    2. **Stance multiplier** — negative events are weighted higher (more market-moving)
    3. **Model confidence** — lower confidence reduces the score

Formula:
    impact_score = base_weight × stance_multiplier × avg_confidence × 10

Usage:
    >>> from models.impact_scorer import ImpactScorer
    >>> scorer = ImpactScorer()
    >>> score = scorer.compute("mergers_acquisitions", "positive", 0.85, 0.90)
    >>> print(score)
    5.95
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity Weights (0.0 – 1.0)
# ---------------------------------------------------------------------------
# Higher = more impactful event category on average.

EVENT_SEVERITY: dict[str, float] = {
    "earnings":              0.80,
    "leadership_change":     0.75,
    "regulatory_action":     0.85,
    "mergers_acquisitions":  1.00,
    "legal_action":          0.70,
    "product_announcement":  0.60,
    "market_movement":       0.90,
}

# ---------------------------------------------------------------------------
# Stance Multipliers
# ---------------------------------------------------------------------------
# Negative news tends to be more market-moving than positive/neutral news.

STANCE_MULTIPLIER: dict[str, float] = {
    "positive": 0.70,
    "negative": 1.00,
    "neutral":  0.40,
}


class ImpactScorer:
    """
    Compute a composite impact score (0–10) for a classified article.

    The score reflects how potentially market-moving an article is, based on
    the event type, sentiment stance, and the model's classification confidence.

    Attributes
    ----------
    event_severity : dict[str, float]
        Base severity weights per event category.
    stance_multiplier : dict[str, float]
        Multipliers per stance label.
    """

    def __init__(
        self,
        event_severity: dict[str, float] | None = None,
        stance_multiplier: dict[str, float] | None = None,
    ):
        self.event_severity = event_severity or EVENT_SEVERITY
        self.stance_multiplier = stance_multiplier or STANCE_MULTIPLIER

    def compute(
        self,
        event_type: str,
        stance: str,
        event_confidence: float,
        stance_confidence: float,
    ) -> float:
        """
        Compute the impact score for a single article.

        Parameters
        ----------
        event_type : str
            Predicted event category (e.g. ``"earnings"``).
        stance : str
            Predicted stance (``"positive"``, ``"negative"``, ``"neutral"``).
        event_confidence : float
            Model confidence for the event prediction (0.0–1.0).
        stance_confidence : float
            Model confidence for the stance prediction (0.0–1.0).

        Returns
        -------
        float
            Impact score in the range 0.0–10.0, rounded to 2 decimals.
        """
        base = self.event_severity.get(event_type.lower(), 0.5)
        multiplier = self.stance_multiplier.get(stance.lower(), 0.5)
        avg_confidence = (event_confidence + stance_confidence) / 2.0

        raw_score = base * multiplier * avg_confidence * 10.0

        # Clamp to [0, 10]
        score = max(0.0, min(10.0, raw_score))
        return round(score, 2)

    def compute_batch(
        self,
        event_types: list[str],
        stances: list[str],
        event_confidences: list[float],
        stance_confidences: list[float],
    ) -> list[float]:
        """
        Compute impact scores for a batch of articles.

        Parameters
        ----------
        event_types : list[str]
        stances : list[str]
        event_confidences : list[float]
        stance_confidences : list[float]

        Returns
        -------
        list[float]
            List of impact scores.
        """
        return [
            self.compute(et, st, ec, sc)
            for et, st, ec, sc in zip(event_types, stances, event_confidences, stance_confidences)
        ]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    scorer = ImpactScorer()
    test_cases = [
        ("earnings",              "positive", 0.92, 0.88),
        ("mergers_acquisitions",  "negative", 0.85, 0.90),
        ("product_announcement",  "neutral",  0.70, 0.65),
        ("regulatory_action",     "negative", 0.95, 0.91),
        ("market_movement",       "positive", 0.60, 0.55),
    ]
    print("\nImpact Scoring Results:")
    print("-" * 60)
    for et, st, ec, sc in test_cases:
        score = scorer.compute(et, st, ec, sc)
        print(f"  Score: {score:5.2f}  ←  {et:25s} / {st:10s}  (conf: {ec:.2f}, {sc:.2f})")

"""
event_taxonomy.py — Financial Event Taxonomy for RNIA
=====================================================

Defines the standard event categories and stance labels used throughout
the Event-Driven Retail News Impact Analyst system.

Event Categories (7 — consolidated from V2 9-class taxonomy):
    1  Earnings
    2  Leadership_Change
    3  Regulatory_Action
    4  Mergers_Acquisitions
    5  Legal_Action
    6  Product_Announcement
    7  Market_Movement

Stance Labels (4 — market-oriented):
    bullish, bearish, neutral, mixed
"""

# ---------------------------------------------------------------------------
# Event Categories
# ---------------------------------------------------------------------------

EVENT_CATEGORIES: dict[int, str] = {
    1: "Earnings",
    2: "Leadership_Change",
    3: "Regulatory_Action",
    4: "Mergers_Acquisitions",
    5: "Legal_Action",
    6: "Product_Announcement",
    7: "Market_Movement",
}

# Legacy 9-class → 7-class collapse mapping (applied at training time)
EVENT_COLLAPSE: dict[str, str] = {
    "Macroeconomic_Geopolitical": "Market_Movement",
    "Market_Sentiment_Investor_Action": "Market_Movement",
    "Other": "Market_Movement",
}

# Canonical lowercase set for quick validation
EVENT_CATEGORY_SET = {v.lower() for v in EVENT_CATEGORIES.values()}

# ---------------------------------------------------------------------------
# Stance Labels (3-class — mixed collapsed into neutral for training)
# ---------------------------------------------------------------------------

STANCE_LABELS: dict[int, str] = {
    1: "bullish",
    2: "bearish",
    3: "neutral",
}

STANCE_LABEL_SET = {v for v in STANCE_LABELS.values()}

# ---------------------------------------------------------------------------
# Human-readable display names (for UI / explanation generator)
# ---------------------------------------------------------------------------

EVENT_DISPLAY_NAMES: dict[str, str] = {
    "Earnings":                          "Earnings & Financial Results",
    "Leadership_Change":                 "Leadership Change",
    "Regulatory_Action":                 "Regulatory Action",
    "Mergers_Acquisitions":              "Mergers & Acquisitions",
    "Legal_Action":                      "Legal Action",
    "Product_Announcement":              "Product Announcement",
    "Market_Movement":                   "Market Movement",
}

# ---------------------------------------------------------------------------
# V2 → Frontend mapping (used by backend/app.py)
# ---------------------------------------------------------------------------

V2_TO_FRONTEND_EVENT = {
    "Earnings":                          "earnings",
    "Mergers_Acquisitions":              "ma",
    "Regulatory_Action":                 "regulatory",
    "Leadership_Change":                 "leadership",
    "Legal_Action":                      "legal",
    "Product_Announcement":              "product",
    "Market_Movement":                   "market",
    # Legacy compat — in case old DB rows still have the 9-class labels
    "Macroeconomic_Geopolitical":        "market",
    "Market_Sentiment_Investor_Action":  "market",
    "Other":                             "market",
    "Unclassified":                      "market",
}

V2_TO_FRONTEND_STANCE = {
    "bullish":  "bullish",
    "bearish":  "bearish",
    "neutral":  "neutral",
    "mixed":    "neutral",   # legacy compat — collapsed at training time
}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def get_event_categories() -> dict[int, str]:
    """
    Return the full event-category mapping.

    Returns
    -------
    dict[int, str]
        Mapping of numeric ID → category label, e.g. {1: "Earnings", …}.

    Example
    -------
    >>> cats = get_event_categories()
    >>> cats[1]
    'Earnings'
    """
    return EVENT_CATEGORIES.copy()


def validate_event_category(label: str) -> bool:
    """
    Check whether *label* is a valid event-category name (case-insensitive).

    Parameters
    ----------
    label : str
        Category label to validate (e.g. "Earnings", "earnings").

    Returns
    -------
    bool
        ``True`` if the label matches one of the defined categories.

    Example
    -------
    >>> validate_event_category('Earnings')
    True
    >>> validate_event_category('Unknown')
    False
    """
    return label.strip().lower() in EVENT_CATEGORY_SET


def validate_stance(label: str) -> bool:
    """
    Check whether *label* is a valid stance label (case-insensitive).

    Parameters
    ----------
    label : str
        Stance label to validate (e.g. "bullish", "Neutral").

    Returns
    -------
    bool
        ``True`` if the label matches one of the defined stance labels.

    Example
    -------
    >>> validate_stance('bullish')
    True
    >>> validate_stance('bad')
    False
    """
    return label.strip().lower() in STANCE_LABEL_SET


def get_stance_labels() -> dict[int, str]:
    """
    Return the full stance-label mapping.

    Returns
    -------
    dict[int, str]
        Mapping of numeric ID → stance label, e.g. {1: "bullish", …}.
    """
    return STANCE_LABELS.copy()


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Event Categories:")
    for num, cat in EVENT_CATEGORIES.items():
        print(f"  {num}  {cat}")

    print("\nStance Labels:")
    for num, stance in STANCE_LABELS.items():
        print(f"  {num}  {stance}")

    # Validation demos
    print("\nValidation tests:")
    print(f"  validate_event_category('Earnings')                    → {validate_event_category('Earnings')}")
    print(f"  validate_event_category('Macroeconomic_Geopolitical')  → {validate_event_category('Macroeconomic_Geopolitical')}")
    print(f"  validate_event_category('unknown')                     → {validate_event_category('unknown')}")
    print(f"  validate_stance('bullish')                             → {validate_stance('bullish')}")
    print(f"  validate_stance('bad')                                 → {validate_stance('bad')}")

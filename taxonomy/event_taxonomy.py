"""
event_taxonomy.py — Financial Event Taxonomy for RNIA
=====================================================

Defines the standard event categories and stance labels used throughout
the Event-Driven Retail News Impact Analyst system.

Event Categories (7):
    1  Earnings
    2  Leadership_Change
    3  Regulatory_Action
    4  Mergers_Acquisitions
    5  Legal_Action
    6  Product_Announcement
    7  Market_Movement

Stance Labels:
    positive, negative, neutral
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

# ---------------------------------------------------------------------------
# Stance Labels
# ---------------------------------------------------------------------------

STANCE_LABELS: dict[int, str] = {
    1: "positive",
    2: "negative",
    3: "neutral",
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
    >>> validate_event_category("Earnings")
    True
    >>> validate_event_category("Unknown")
    False
    """
    valid = {v.lower() for v in EVENT_CATEGORIES.values()}
    return label.strip().lower() in valid


def validate_stance(label: str) -> bool:
    """
    Check whether *label* is a valid stance label (case-insensitive).

    Parameters
    ----------
    label : str
        Stance label to validate (e.g. "positive", "Neutral").

    Returns
    -------
    bool
        ``True`` if the label matches one of the defined stance labels.

    Example
    -------
    >>> validate_stance("positive")
    True
    >>> validate_stance("unknown")
    False
    """
    valid = {v.lower() for v in STANCE_LABELS.values()}
    return label.strip().lower() in valid


def get_stance_labels() -> dict[int, str]:
    """
    Return the full stance-label mapping.

    Returns
    -------
    dict[int, str]
        Mapping of numeric ID → stance label, e.g. {1: "positive", …}.
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
    print(f"  validate_event_category('Earnings')    → {validate_event_category('Earnings')}")
    print(f"  validate_event_category('unknown')      → {validate_event_category('unknown')}")
    print(f"  validate_stance('positive')              → {validate_stance('positive')}")
    print(f"  validate_stance('bad')                   → {validate_stance('bad')}")

"""
annotation_tool.py — CLI Annotation Tool for RNIA
==================================================

Interactive command-line tool that lets a human annotator label each
news article with an **event type** and a **stance**.

Features:
    • Reads the cleaned dataset from
      ``data/processed_news/news_clean_dataset.csv``
    • Displays headline + first 300 characters of text for each article
    • Prompts for event type (1–7) and stance (1–3)
    • Saves each labeled row **immediately** to
      ``data/labeled_dataset/financial_news_labeled.csv``
    • Supports **stop & resume** — already-labeled URLs are skipped
    • Enter ``q`` at any prompt to quit gracefully

Output Columns:
    headline, clean_text, event_type, stance, source, timestamp, url
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# We add the project root to sys.path so we can import the taxonomy module
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from taxonomy.event_taxonomy import EVENT_CATEGORIES, STANCE_LABELS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "labeled_dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "financial_news_labeled.csv")

# Columns for the labeled dataset
OUTPUT_COLUMNS = ["headline", "clean_text", "event_type", "stance", "source", "timestamp", "url"]

# How many characters of clean_text to display
PREVIEW_LENGTH = 300


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def load_clean_dataset() -> pd.DataFrame:
    """
    Load the cleaned news dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least the columns required for labeling.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset CSV does not exist.
    """
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            f"Cleaned dataset not found at:\n  {INPUT_FILE}\n"
            "Please run preprocessing/clean_text.py first."
        )
    return pd.read_csv(INPUT_FILE, encoding="utf-8-sig")


def load_labeled_urls() -> set[str]:
    """
    Load URLs that have already been labeled (for resume support).

    Returns
    -------
    set[str]
        Set of URLs already present in the labeled output file.
    """
    if os.path.isfile(OUTPUT_FILE):
        try:
            labeled_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
            return set(labeled_df["url"].dropna().astype(str))
        except Exception:
            return set()
    return set()


def append_labeled_row(row: dict) -> None:
    """
    Append a single labeled row to the output CSV.

    If the file does not exist yet, it is created with the header row.
    Otherwise the row is appended without repeating the header.

    Parameters
    ----------
    row : dict
        Dictionary with keys matching ``OUTPUT_COLUMNS``.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_header = not os.path.isfile(OUTPUT_FILE)

    row_df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
    row_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def prompt_choice(prompt_text: str, valid_range: range) -> int | None:
    """
    Prompt the user for a numeric choice within *valid_range*.

    Parameters
    ----------
    prompt_text : str
        Message shown to the user.
    valid_range : range
        Acceptable integer range (e.g. ``range(1, 8)`` for 1–7).

    Returns
    -------
    int or None
        The selected number, or ``None`` if the user entered ``q`` to quit.
    """
    while True:
        answer = input(prompt_text).strip()
        if answer.lower() == "q":
            return None
        try:
            choice = int(answer)
            if choice in valid_range:
                return choice
            print(f"  ⚠  Please enter a number between {valid_range.start} and {valid_range.stop - 1}.")
        except ValueError:
            print("  ⚠  Invalid input. Enter a number or 'q' to quit.")


def display_event_menu() -> None:
    """Print the event-type selection menu."""
    print("\n  EVENT TYPE:")
    for num, label in EVENT_CATEGORIES.items():
        print(f"    {num}  {label}")


def display_stance_menu() -> None:
    """Print the stance selection menu."""
    print("\n  STANCE:")
    for num, label in STANCE_LABELS.items():
        print(f"    {num}  {label.capitalize()}")


# ---------------------------------------------------------------------------
# Main Annotation Loop
# ---------------------------------------------------------------------------


def main():
    """Run the interactive annotation session."""
    print("=" * 64)
    print("  RNIA — Article Annotation Tool")
    print("=" * 64)
    print("  Enter 'q' at any prompt to save progress and quit.\n")

    # Load data
    df = load_clean_dataset()
    labeled_urls = load_labeled_urls()

    # Filter out already-labeled articles
    remaining = df[~df["url"].astype(str).isin(labeled_urls)].reset_index(drop=True)
    total = len(df)
    done = total - len(remaining)

    if remaining.empty:
        print("✅ All articles have been labeled!")
        print(f"   Output: {OUTPUT_FILE}")
        return

    print(f"  Total articles : {total}")
    print(f"  Already labeled: {done}")
    print(f"  Remaining      : {len(remaining)}\n")

    labeled_count = 0

    for idx, row in remaining.iterrows():
        current = done + labeled_count + 1
        print("-" * 64)
        print(f"  [{current}/{total}]")
        print(f"  HEADLINE: {row['headline']}")

        # Preview of the cleaned text
        text_preview = str(row["clean_text"])[:PREVIEW_LENGTH]
        print(f"\n  TEXT PREVIEW:\n  {text_preview}{'…' if len(str(row['clean_text'])) > PREVIEW_LENGTH else ''}")

        # --- Event Type ---
        display_event_menu()
        event_choice = prompt_choice("\n  Select event type (1-7) or 'q': ", range(1, 8))
        if event_choice is None:
            break

        # --- Stance ---
        display_stance_menu()
        stance_choice = prompt_choice("\n  Select stance (1-3) or 'q': ", range(1, 4))
        if stance_choice is None:
            break

        # Build labeled row and save immediately
        labeled_row = {
            "headline": row["headline"],
            "clean_text": row["clean_text"],
            "event_type": EVENT_CATEGORIES[event_choice].lower(),
            "stance": STANCE_LABELS[stance_choice],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "url": row["url"],
        }
        append_labeled_row(labeled_row)
        labeled_count += 1
        print(f"\n  ✓ Labeled as: {labeled_row['event_type']} / {labeled_row['stance']}")

    # --- Session summary ---
    print("\n" + "=" * 64)
    print(f"  Session complete — {labeled_count} article(s) labeled this session.")
    print(f"  Total labeled so far: {done + labeled_count}/{total}")
    print(f"  Output file: {OUTPUT_FILE}")
    print("=" * 64)


if __name__ == "__main__":
    main()

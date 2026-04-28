"""
rebuild_datasets.py — Rebuild cleaned, labeled, and impact-scored datasets
from the raw news dataset.

Runs the three pipeline stages in sequence:
  1. Cleaning   → data/processed_news/news_clean_dataset.csv
  2. Labeling   → data/labeled_dataset/financial_news_labeled.csv
  3. Impact     → data/processed_news/news_with_impact_scores.csv
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    print("=" * 60)
    print("REBUILD — Regenerating all datasets from raw news")
    print("=" * 60)

    # --- Step 1: Cleaning ---
    print("\n[1/3] Running text preprocessing (cleaning)...")
    from preprocessing.clean_text import main as clean_main
    clean_main()

    clean_path = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv")
    if os.path.isfile(clean_path):
        import pandas as pd
        df = pd.read_csv(clean_path)
        print(f"  ✅ Clean dataset: {len(df)} articles")
    else:
        print("  ❌ Clean dataset NOT created. Aborting.")
        return

    # --- Step 2: Labeling ---
    print("\n[2/3] Running auto-labeling (event_type + stance)...")
    from annotation.auto_label_dataset import auto_label_dataset
    auto_label_dataset()

    labeled_path = os.path.join(PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv")
    if os.path.isfile(labeled_path):
        df_labeled = pd.read_csv(labeled_path)
        print(f"  ✅ Labeled dataset: {len(df_labeled)} articles")
    else:
        print("  ❌ Labeled dataset NOT created. Aborting.")
        return

    # --- Step 3: Impact Scoring ---
    print("\n[3/3] Running impact scoring...")
    # The impact scorer needs event_type column, so we feed it the labeled dataset
    # but save to the impact scores path
    from scoring.impact_score import process_dataset
    impact_input = labeled_path  # Use labeled dataset (has event_type)
    impact_output = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_with_impact_scores.csv")
    process_dataset(input_csv=impact_input, output_csv=impact_output)

    if os.path.isfile(impact_output):
        df_impact = pd.read_csv(impact_output)
        print(f"  ✅ Impact scores dataset: {len(df_impact)} articles")
    else:
        print("  ❌ Impact scores dataset NOT created.")
        return

    # --- Summary ---
    print("\n" + "=" * 60)
    print("✅ ALL DATASETS REBUILT SUCCESSFULLY!")
    print(f"  Clean dataset:   {len(df)} articles  → {clean_path}")
    print(f"  Labeled dataset: {len(df_labeled)} articles  → {labeled_path}")
    print(f"  Impact scores:   {len(df_impact)} articles  → {impact_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

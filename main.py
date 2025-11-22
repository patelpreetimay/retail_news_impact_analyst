"""
main.py — RNIA Main System Pipeline Controller
================================================

Orchestrates the entire Event-Driven Retail News Impact Analyst workflow.

Pipeline Stages:
    1. News collection          — Scrape articles from RSS feeds
    2. Data preprocessing       — Clean and normalize text
    3. Dataset annotation       — (Manual step via CLI tool)
    4. Model training           — Train TF-IDF + Logistic Regression models
    5. News event prediction    — Classify events and detect stance
    6. Impact score computation — Calculate credibility, recency, materiality
    7. Explanation generation   — Generate human-readable explanations
    8. Dashboard launch         — Open Streamlit interactive dashboard

Usage:
    python main.py

    A CLI menu will appear with options for each pipeline stage.
"""

import os
import sys
import subprocess
import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
RAW_DATASET = os.path.join(PROJECT_ROOT, "data", "raw_news", "news_raw_dataset.csv")
CLEAN_DATASET = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_clean_dataset.csv")
LABELED_DATASET = os.path.join(PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv")
SCORED_DATASET = os.path.join(PROJECT_ROOT, "data", "processed_news", "news_with_impact_scores.csv")
FINAL_REPORT = os.path.join(PROJECT_ROOT, "data", "final_outputs", "news_analysis_report.csv")


# ===========================================================================
# STEP 4 — PIPELINE FUNCTIONS
# ===========================================================================


def run_news_collection():
    """
    Stage 1 — Collect financial news articles from RSS feeds.

    Calls ``scrapers.news_scraper.main()`` which:
        - Fetches entries from Reuters, Yahoo Finance, CNBC
        - Downloads full article text
        - Removes duplicates
        - Saves to ``data/raw_news/news_raw_dataset.csv``
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — News Collection")
    print("=" * 60)

    from scrapers.news_scraper import main as scraper_main
    scraper_main()

    if os.path.isfile(RAW_DATASET):
        df = pd.read_csv(RAW_DATASET)
        print(f"\n✅ Collected {len(df)} articles → {RAW_DATASET}")
    else:
        print("⚠ No output file generated. Check scraper logs.")


def run_preprocessing():
    """
    Stage 2 — Clean and normalize the raw text data.

    Calls ``preprocessing.clean_text.main()`` which:
        - Strips HTML tags
        - Lowercases text
        - Collapses whitespace
        - Saves to ``data/processed_news/news_clean_dataset.csv``
    """
    print("\n" + "=" * 60)
    print("STAGE 2 — Text Preprocessing")
    print("=" * 60)

    if not os.path.isfile(RAW_DATASET):
        print("⚠ Raw dataset not found. Run Stage 1 (News Collection) first.")
        return

    from preprocessing.clean_text import main as preprocess_main
    preprocess_main()

    if os.path.isfile(CLEAN_DATASET):
        df = pd.read_csv(CLEAN_DATASET)
        print(f"\n✅ Preprocessed {len(df)} articles → {CLEAN_DATASET}")
    else:
        print("⚠ No output file generated. Check preprocessing logs.")


def run_annotation():
    """
    Stage 3 — Launch the annotation tool for manual dataset labeling.

    This is an interactive CLI tool that lets you label articles with:
        - Event type (e.g. earnings, mergers_acquisitions)
        - Stance (positive, negative, neutral)
    """
    print("\n" + "=" * 60)
    print("STAGE 3 — Dataset Annotation (Interactive)")
    print("=" * 60)
    print("Launching CLI annotation tool...")
    print("Follow the prompts to label articles.\n")

    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "annotation", "annotation_tool.py")])

    if os.path.isfile(LABELED_DATASET):
        df = pd.read_csv(LABELED_DATASET)
        print(f"\n✅ Labeled dataset has {len(df)} articles → {LABELED_DATASET}")


def run_model_training():
    """
    Stage 4 — Train the event classifier and stance detector.

    Calls ``models.train_models.train_and_evaluate()`` which:
        - Loads labeled dataset
        - Splits 80/20 (or trains on all data if < 10 samples)
        - Trains TF-IDF + Logistic Regression models
        - Evaluates and prints classification reports
        - Saves models to ``models/saved_models/``
    """
    print("\n" + "=" * 60)
    print("STAGE 4 — Model Training")
    print("=" * 60)

    if not os.path.isfile(LABELED_DATASET):
        print("⚠ Labeled dataset not found. Run Stage 3 (Annotation) first.")
        return

    from models.train_models import train_and_evaluate
    result = train_and_evaluate()

    if result:
        print("\n✅ Models trained and saved to models/saved_models/")
    else:
        print("⚠ Training failed. Check logs above.")


def run_prediction_pipeline():
    """
    Stage 5 — Run event classification and stance detection on the
    cleaned dataset, then compute impact scores and generate explanations.

    This is the main analysis pipeline that:
        1. Loads trained models
        2. Predicts event_type and stance for each article
        3. Calculates impact scores (credibility, recency, materiality)
        4. Generates human-readable explanations
        5. Saves the final report to ``data/final_outputs/news_analysis_report.csv``
    """
    print("\n" + "=" * 60)
    print("STAGE 5 — Full Analysis Pipeline")
    print("=" * 60)

    # Check prerequisites
    if not os.path.isfile(CLEAN_DATASET):
        print("⚠ Cleaned dataset not found. Run Stages 1-2 first.")
        return

    model_dir = os.path.join(PROJECT_ROOT, "models", "saved_models")
    required_models = ["event_classifier.pkl", "stance_detector.pkl", "tfidf_vectorizer.pkl"]
    missing = [m for m in required_models if not os.path.isfile(os.path.join(model_dir, m))]
    if missing:
        print(f"⚠ Missing trained models: {missing}")
        print("  Run Stage 4 (Model Training) first.")
        return

    # Load models
    print("\n[1/4] Loading trained models...")
    from models.event_classifier import EventClassifierLR
    from models.stance_detector import StanceDetectorLR

    event_clf = EventClassifierLR()
    event_clf.load_model()

    stance_det = StanceDetectorLR()
    stance_det.load_model()

    # Load cleaned dataset
    print("[2/4] Loading cleaned dataset...")
    df = pd.read_csv(CLEAN_DATASET, encoding="utf-8-sig")
    print(f"  Loaded {len(df)} articles.")

    # Predict event_type and stance
    print("[3/4] Predicting event types and stances...")
    event_types = []
    stances = []
    for idx, row in df.iterrows():
        text = str(row.get("clean_text", ""))
        event_types.append(event_clf.predict_event(text))
        stances.append(stance_det.predict_stance(text))

    df["event_type"] = event_types
    df["stance"] = stances
    print(f"  Predictions complete for {len(df)} articles.")

    # Compute impact scores
    print("[4/4] Computing impact scores and generating explanations...")
    from scoring.impact_score import calculate_impact_score
    from reporting.explanation_generator import generate_explanation

    impact_scores = []
    credibilities = []
    recencies = []
    materialities = []
    explanations = []

    for idx, row in df.iterrows():
        # Impact score
        score_result = calculate_impact_score(
            source=str(row.get("source", "")),
            timestamp=str(row.get("timestamp", "")),
            event_type=str(row.get("event_type", "market_movement")),
        )
        impact_scores.append(score_result["impact_score"])
        credibilities.append(score_result["credibility"])
        recencies.append(score_result["recency"])
        materialities.append(score_result["materiality"])

        # Explanation
        expl_result = generate_explanation(
            event_type=str(row.get("event_type", "")),
            stance=str(row.get("stance", "")),
            impact_score=score_result["impact_score"],
            credibility=score_result["credibility"],
            recency=score_result["recency"],
            materiality=score_result["materiality"],
        )
        explanations.append(expl_result["explanation"])

    # Add columns to DataFrame
    df["credibility"] = credibilities
    df["recency"] = recencies
    df["materiality"] = materialities
    df["impact_score"] = impact_scores
    df["explanation"] = explanations

    # Save final report
    os.makedirs(os.path.dirname(FINAL_REPORT), exist_ok=True)
    df.to_csv(FINAL_REPORT, index=False, encoding="utf-8-sig")

    print(f"\n✅ Analysis complete!")
    print(f"   Articles processed  : {len(df)}")
    print(f"   Avg impact score    : {df['impact_score'].mean():.4f}")
    print(f"   Output saved to     : {FINAL_REPORT}")

    # Also save the scored dataset for the scoring module
    scored_dir = os.path.dirname(SCORED_DATASET)
    os.makedirs(scored_dir, exist_ok=True)
    df.to_csv(SCORED_DATASET, index=False, encoding="utf-8-sig")

    return df


def run_evaluation():
    """
    Stage 6 — Evaluate model performance with metrics and confusion matrices.

    Runs ``evaluation.evaluate_models.evaluate_models()`` which:
        - Loads labeled data and trained models
        - Computes accuracy, precision, recall, F1
        - Generates confusion matrix plots
        - Saves results to ``data/evaluation_results/``
    """
    print("\n" + "=" * 60)
    print("STAGE 6 — Model Evaluation")
    print("=" * 60)

    if not os.path.isfile(LABELED_DATASET):
        print("⚠ Labeled dataset not found. Run Stage 3 (Annotation) first.")
        return

    from evaluation.evaluate_models import evaluate_models
    evaluate_models()


def launch_dashboard():
    """
    Stage 7 — Launch the Streamlit interactive dashboard.

    Runs ``streamlit run dashboard/app.py`` which opens a browser
    with the full analysis results visualization.
    """
    print("\n" + "=" * 60)
    print("STAGE 7 — Launching Streamlit Dashboard")
    print("=" * 60)

    if not os.path.isfile(FINAL_REPORT):
        print("⚠ Final report not found. Run Stage 5 (Analysis Pipeline) first.")
        return

    app_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    print(f"Starting dashboard: streamlit run {app_path}")
    print("Press Ctrl+C to stop the dashboard.\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "true",
    ])


# ===========================================================================
# STEP 8 — USER MENU
# ===========================================================================

def show_menu():
    """Display the interactive CLI menu and return the user's choice."""
    print("\n" + "=" * 60)
    print("  📰 RNIA — Retail News Impact Analyst")
    print("  Event-Driven Financial News Analysis System")
    print("=" * 60)
    print()
    print("  Pipeline Stages:")
    print("  ─────────────────────────────────────────")
    print("  [1]  Collect news articles (RSS scraping)")
    print("  [2]  Preprocess dataset (text cleaning)")
    print("  [3]  Annotate dataset (manual labeling)")
    print("  [4]  Train ML models")
    print("  [5]  Run analysis pipeline (predict + score + explain)")
    print("  [6]  Run model evaluation")
    print("  [7]  Launch Streamlit dashboard")
    print("  ─────────────────────────────────────────")
    print("  [8]  Run full pipeline (Stages 1→2→5)")
    print("  [0]  Exit")
    print()

    choice = input("  Enter your choice [0-8]: ").strip()
    return choice


def run_full_pipeline():
    """
    Run the complete automated pipeline (Stages 1 → 2 → 5).

    Skips annotation (Stage 3) and model training (Stage 4) since
    these require prior manual steps. Uses pre-trained models.
    """
    print("\n" + "🔁" * 30)
    print("RUNNING FULL AUTOMATED PIPELINE")
    print("🔁" * 30)

    run_news_collection()
    run_preprocessing()
    run_prediction_pipeline()

    print("\n" + "=" * 60)
    print("✅ Full pipeline complete!")
    print(f"   Final report: {FINAL_REPORT}")
    print("   Launch dashboard with option [7] to explore results.")
    print("=" * 60)


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":
    # Menu-driven pipeline controller
    while True:
        choice = show_menu()

        if choice == "1":
            run_news_collection()
        elif choice == "2":
            run_preprocessing()
        elif choice == "3":
            run_annotation()
        elif choice == "4":
            run_model_training()
        elif choice == "5":
            run_prediction_pipeline()
        elif choice == "6":
            run_evaluation()
        elif choice == "7":
            launch_dashboard()
        elif choice == "8":
            run_full_pipeline()
        elif choice == "0":
            print("\nGoodbye! 👋")
            sys.exit(0)
        else:
            print("\n⚠ Invalid choice. Please enter a number from 0 to 8.")

        input("\nPress Enter to return to the menu...")

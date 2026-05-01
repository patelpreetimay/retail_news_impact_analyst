"""
main.py — RNIA Main System Pipeline Controller
================================================

Orchestrates the Event-Driven Retail News Impact Analyst workflow.

ML-only runtime — Gemini was used ONCE to label the seed dataset and is
never called at runtime.

Stages:
    1. Scrape + preprocess (RSS → cleaned CSV) and ingest into rnia.db
    2. Train relevance classifier (binary; on full ~5,400-row dataset)
    3. Train classifier suite (event + stance + 4 sub-score regressors;
       on the ~1,800 high-quality rel=1 subset)
    4. Run pipeline on new articles
       (deterministic keyword filter → ML relevance → event/stance/impact
        → escalation safety check → grow datasets)
    5. Launch web dashboard (FastAPI backend; start React separately)
    6. Run model evaluation
    7. Run full pipeline (1 → 2 → 3 → 4)
    8. Run full system check

See docs/pipeline-algorithm-spec.md for the canonical algorithm.

Usage:
    python main.py
"""

import io
import os
import sys
import subprocess
import logging
import sqlite3

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
# Fix Windows console encoding for emoji / Unicode
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_NEWS_CSV = os.path.join(DATA_DIR, "raw_news", "news_raw_dataset.csv")
CLEAN_NEWS_CSV = os.path.join(DATA_DIR, "processed_news", "news_clean_dataset.csv")
LABELED_CSV = os.path.join(DATA_DIR, "labeled_dataset", "financial_news_labeled.csv")
REPORT_CSV = os.path.join(DATA_DIR, "final_outputs", "news_analysis_report.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "rnia.db")

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")

# Models required before Stage 4 (classification) can run.
_REQUIRED_MODELS = (
    "relevance_classifier.pkl", "tfidf_relevance_vectorizer.pkl",
    "event_classifier.pkl", "tfidf_vectorizer.pkl",
    "stance_detector.pkl", "tfidf_stance_vectorizer.pkl",
    "tfidf_subscore_vectorizer.pkl",
)


def _models_present() -> bool:
    """True iff every model needed by run() is on disk."""
    return all(
        os.path.isfile(os.path.join(SAVED_MODELS_DIR, m))
        for m in _REQUIRED_MODELS
    )


def _count_relevant_done() -> int:
    """rel=1 done rows in v2_analyses — exactly what the dashboard shows."""
    return _snapshot_state().get("rel1_done", 0)


_RELEVANCE_CSV = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "relevance_dataset.csv"
)


def _snapshot_state() -> dict:
    """
    Capture every count the pipeline summary cares about, in one shot.
    Diff two snapshots to get a 'this run' breakdown.
    """
    state = {
        "articles_total":   0,
        "rel1_done":        0,  # visible on dashboard
        "rel1_escalation":  0,  # rel=1 but flagged low-confidence
        "rel0_total":       0,
        "filtered":         0,  # subset of rel0: rejected by keyword/junk filter
        "dataset1_rows":    0,  # rows in relevance_dataset.csv
    }
    if os.path.isfile(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            state["articles_total"] = cur.execute(
                "SELECT COUNT(*) FROM articles"
            ).fetchone()[0]
            state["rel1_done"] = cur.execute(
                "SELECT COUNT(*) FROM v2_analyses "
                "WHERE relevance=1 AND processing_status='done'"
            ).fetchone()[0]
            state["rel1_escalation"] = cur.execute(
                "SELECT COUNT(*) FROM v2_analyses "
                "WHERE relevance=1 AND processing_status='needs_escalation'"
            ).fetchone()[0]
            state["rel0_total"] = cur.execute(
                "SELECT COUNT(*) FROM v2_analyses WHERE relevance=0"
            ).fetchone()[0]
            state["filtered"] = cur.execute(
                "SELECT COUNT(*) FROM v2_analyses "
                "WHERE processing_status='filtered'"
            ).fetchone()[0]
            conn.close()
        except Exception:
            pass
    if os.path.isfile(_RELEVANCE_CSV):
        try:
            import pandas as pd
            df = pd.read_csv(_RELEVANCE_CSV, encoding="utf-8-sig")
            state["dataset1_rows"] = len(df)
        except Exception:
            pass
    return state


def _fmt_delta(before: int, after: int) -> str:
    d = after - before
    return f"{before:>6d} → {after:<6d}  ({'+' if d >= 0 else ''}{d})"


def _print_pipeline_summary(before: dict, after: dict, title: str = "DATASETS SUMMARY (this run)") -> None:
    """Comprehensive before/after block for Stage 1, Stage 4, full pipeline."""
    new_rel1_done = after["rel1_done"]       - before["rel1_done"]
    new_rel1_esc  = after["rel1_escalation"] - before["rel1_escalation"]
    new_rel0      = after["rel0_total"]      - before["rel0_total"]
    new_filtered  = after["filtered"]        - before["filtered"]
    new_rel1      = new_rel1_done + new_rel1_esc
    new_ml_rej    = new_rel0 - new_filtered

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  Articles in DB:              {_fmt_delta(before['articles_total'], after['articles_total'])}")
    print()
    print("  Dataset 1  (rel=0 + rel=1, used for retraining)")
    print(f"    relevance_dataset.csv:     {_fmt_delta(before['dataset1_rows'], after['dataset1_rows'])}")
    print()
    print("  Dataset 2  (rel=1 + classification — shown on web UI)")
    print(f"    visible on dashboard:      {_fmt_delta(before['rel1_done'], after['rel1_done'])}")
    print(f"    hidden (needs_escalation): {_fmt_delta(before['rel1_escalation'], after['rel1_escalation'])}")
    print()
    print("  This run's classifier output:")
    print(f"    relevance = 1               +{new_rel1}")
    print(f"      → added to web UI         +{new_rel1_done}")
    print(f"      → flagged needs_escalation +{new_rel1_esc}")
    print(f"    relevance = 0               +{new_rel0}")
    print(f"      → keyword/junk filter     +{new_filtered}")
    print(f"      → ML rejected             +{new_ml_rej}")
    print("=" * 60)


# ===========================================================================
# STAGE 1 — Scrape + Preprocess + Ingest into DB
# ===========================================================================

def stage_1_scrape_and_preprocess():
    """
    Stage 1 — Scrape RSS feeds → clean text → ingest → classify.

    Pipeline:
        a. Scrape news from 30+ RSS feeds (saves news_raw_dataset.csv)
        b. Clean text (saves news_clean_dataset.csv)
        c. Ingest cleaned rows into `articles` (URL UNIQUE → idempotent)
        d. Run classifier on the freshly-ingested article_ids so they
           land in v2_analyses and the rel=1 ones show up on the dashboard.
           Skipped (with a clear message) if models aren't trained yet.
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — Scrape + Preprocess + Ingest + Classify")
    print("=" * 60)

    state_before = _snapshot_state()

    # 1a — Scrape
    from scrapers.news_scraper import main as scraper_main
    print("\n  [a] Scraping RSS feeds...")
    scraper_main()
    if not os.path.isfile(RAW_NEWS_CSV):
        print("  ⚠ Raw dataset was not created. Aborting.")
        return False
    import pandas as pd
    raw_df = pd.read_csv(RAW_NEWS_CSV, encoding="utf-8-sig")
    print(f"      Raw dataset:     {len(raw_df)} articles")

    # 1b — Clean
    from preprocessing.clean_text import main as clean_main
    print("\n  [b] Cleaning text...")
    clean_main()
    if not os.path.isfile(CLEAN_NEWS_CSV):
        print("  ⚠ Cleaned dataset was not created. Aborting.")
        return False
    clean_df = pd.read_csv(CLEAN_NEWS_CSV, encoding="utf-8-sig")
    print(f"      Cleaned dataset: {len(clean_df)} articles")

    # 1c — Ingest into DB
    from pipeline.ingest import ingest_clean_csv
    print("\n  [c] Ingesting into data/rnia.db...")
    new_ids = ingest_clean_csv()
    print(f"      Newly inserted:  {len(new_ids)} article(s)")
    if new_ids:
        print(f"      ID range:        {min(new_ids)} – {max(new_ids)}")

    # 1d — Classify the new articles so the dashboard's rel=1 set grows
    if new_ids:
        if _models_present():
            print(f"\n  [d] Classifying {len(new_ids)} new article(s)...")
            from pipeline.run import run
            summary = run(article_ids=new_ids)
            for k, v in summary.items():
                print(f"      {k:22s} {v}")
        else:
            missing = [
                m for m in _REQUIRED_MODELS
                if not os.path.isfile(os.path.join(SAVED_MODELS_DIR, m))
            ]
            print("\n  [d] Skipping classification — models not trained yet.")
            print(f"      Missing: {missing}")
            print("      Train with [2] then [3], then re-run [4] to classify.")
    else:
        print("\n  [d] No new articles to classify.")

    _print_pipeline_summary(
        state_before, _snapshot_state(),
        title="STAGE 1 SUMMARY",
    )
    return True


# ===========================================================================
# STAGE 2 — Train Relevance Classifier
# ===========================================================================

def stage_2_train_relevance():
    """
    Stage 2 — (Re-)train the binary relevance classifier on the
    full ~5,400-row relevance dataset (rel=0 + rel=1).

    Steps:
        a. Refresh data/labeled_dataset/relevance_dataset.csv from rnia.db
        b. Train TF-IDF + Logistic Regression on it; save models
    """
    print("\n" + "=" * 60)
    print("STAGE 2 — Train Relevance Classifier")
    print("=" * 60)

    print("\n  [a] Refreshing relevance dataset from DB...")
    export_script = os.path.join(
        PROJECT_ROOT, "scripts", "export_relevance_dataset.py"
    )
    subprocess.run([sys.executable, export_script], check=False)

    print("\n  [b] Training relevance classifier...")
    from models.train_models import train_relevance_model
    clf = train_relevance_model()
    if clf is None:
        print("\n  ⚠ Relevance training failed.")
        return False
    print("\n  ✅ Relevance classifier trained.")
    return True


# ===========================================================================
# STAGE 3 — Train Classifier Suite (event + stance + sub-scores)
# ===========================================================================

def stage_3_train_classifiers():
    """
    Stage 3 — Train the three label models on the gold subset
    (relevance=1 only), then the four sub-score regressors that feed
    the deterministic impact formula.

    Models produced:
        event_classifier.pkl      (TF-IDF + LogReg; 7 event types)
        stance_detector.pkl       (TF-IDF + LogReg; 3 stances)
        subscore_*.pkl × 4        (Ridge regressors per sub-score)
    """
    print("\n" + "=" * 60)
    print("STAGE 3 — Train Classifier Suite (event / stance / sub-scores)")
    print("=" * 60)

    print("\n  [a] Refreshing gold dataset from DB...")
    export_script = os.path.join(
        PROJECT_ROOT, "scripts", "export_gold_dataset.py"
    )
    subprocess.run([sys.executable, export_script], check=False)

    print("\n  [b] Training event classifier + stance detector...")
    from models.train_models import train_and_evaluate, train_sub_score_models
    result = train_and_evaluate()
    if result is None:
        print("\n  ⚠ event/stance training failed.")
        return False

    print("\n  [c] Training sub-score regressors...")
    sub = train_sub_score_models()
    if sub is None:
        print("\n  ⚠ sub-score training failed.")
        return False

    print("\n  ✅ Classifier suite trained.")
    return True


# ===========================================================================
# STAGE 4 — Run Pipeline on New Articles
# ===========================================================================

def stage_4_run():
    """
    Stage 4 — Process every article in the DB that has no v2_analyses
    row yet.

    For each new article:
        1. Deterministic keyword filter (junk_stub / body_too_short /
           no_keyword_match) → if rejected, write filtered row.
        2. ML relevance classifier → if rel=0, write done row (hidden
           from dashboard but kept for retraining).
        3. event/stance/sub-score predictions; deterministic impact_score
           via the 0.4·M + 0.3·L + 0.2·T + 0.1·C formula.
        4. Escalation safety check (low confidence / inconsistent
           sub-scores / high-stakes BEARISH on regulatory/legal).
        5. Append rows to relevance_dataset.csv (always) and gold CSV
           (only when min(event_conf, stance_conf) ≥ 0.85, no escalation).
    """
    print("\n" + "=" * 60)
    print("STAGE 4 — Run Pipeline (predict + score + safety check)")
    print("=" * 60)

    if not _models_present():
        missing = [
            m for m in _REQUIRED_MODELS
            if not os.path.isfile(os.path.join(SAVED_MODELS_DIR, m))
        ]
        print(f"  ⚠ Missing models: {missing}")
        print("     Run Stage 2 and Stage 3 first.")
        return False

    state_before = _snapshot_state()

    from pipeline.run import run
    summary = run()

    print("\n  pipeline counters:")
    for k, v in summary.items():
        print(f"    {k:22s} {v}")

    _print_pipeline_summary(
        state_before, _snapshot_state(),
        title="STAGE 4 SUMMARY",
    )

    if summary.get("processed", 0) == 0:
        print("\n  (No unprocessed articles found — run Stage 1 to ingest more.)")
    else:
        print("\n  ✅ Pipeline complete.")
    return True


# ===========================================================================
# STAGE 6 — Model Evaluation
# ===========================================================================

def stage_6_evaluation():
    """Run model evaluation reports."""
    print("\n" + "=" * 60)
    print("STAGE 6 — Model Evaluation")
    print("=" * 60)

    eval_script = os.path.join(PROJECT_ROOT, "evaluation", "evaluate_models.py")
    if os.path.isfile(eval_script):
        subprocess.run([sys.executable, eval_script])
    else:
        print("  Running quick evaluation via training pipeline...")
        from models.train_models import train_and_evaluate
        train_and_evaluate()
    return True


# ===========================================================================
# STAGE 9 — FinBERT Comparison Baseline
# ===========================================================================

def stage_9_finbert_comparison():
    """Run FinBERT zero-shot comparison against the LR stance detector."""
    print("\n" + "=" * 60)
    print("STAGE 9 — FinBERT Comparison Baseline")
    print("=" * 60)
    print("  First run downloads ~440 MB from HuggingFace (one-time, then cached).")
    print("  Requires: pip install transformers torch")
    print()

    script = os.path.join(PROJECT_ROOT, "evaluation", "finbert_comparison.py")
    if os.path.isfile(script):
        subprocess.run([sys.executable, script])
    else:
        print("  ⚠ evaluation/finbert_comparison.py not found.")
        return False
    return True


# ===========================================================================
# STAGE 7 — Launch Web Dashboard
# ===========================================================================

def stage_7_dashboard():
    """
    Launch the RNIA Web Dashboard (FastAPI + React).

    Starts the FastAPI backend server. The React frontend should be
    started separately with `npm run dev` inside the frontend/ directory.
    """
    print("\n" + "=" * 60)
    print("LAUNCHING RNIA WEB DASHBOARD")
    print("=" * 60)

    # Quick DB health check
    if os.path.isfile(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            done_count = conn.execute(
                "SELECT COUNT(*) FROM v2_analyses WHERE processing_status = 'done' AND relevance = 1"
            ).fetchone()[0]
            total_articles = conn.execute(
                "SELECT COUNT(*) FROM articles"
            ).fetchone()[0]
            conn.close()
            print(f"\n  Database: {DB_PATH}")
            print(f"  Total articles in DB  : {total_articles}")
            print(f"  Analysed & relevant   : {done_count}")
        except Exception as exc:
            print(f"  ⚠ DB check warning: {exc}")
    else:
        print(f"  ⚠ Database not found: {DB_PATH}")
        print("     Run the pipeline first (option 8).")

    print("\n  Starting FastAPI backend server...")
    print("  API will be available at: http://127.0.0.1:8000")
    print("  Swagger docs at:          http://127.0.0.1:8000/docs")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  To start the React frontend, open a NEW terminal:  │")
    print("  │                                                     │")
    print("  │    cd frontend                                      │")
    print("  │    npm run dev                                      │")
    print("  │                                                     │")
    print("  │  Then open: http://localhost:5173                    │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Press Ctrl+C to stop the backend server.\n")

    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.app:app", "--reload",
        "--host", "127.0.0.1", "--port", "8000",
    ])


# ===========================================================================
# FULL PIPELINE — Stages 1→2→3→4→5
# ===========================================================================

def run_full_pipeline():
    """
    Run the complete pipeline (Stages 1 → 2 → 3 → 4).

    1. Scrape + preprocess + ingest into rnia.db
    2. Train relevance classifier on the full ~5,400-row dataset
    3. Train classifier suite (event + stance + sub-scores) on the gold subset
    4. Run pipeline (deterministic filter → ML relevance → event/stance/impact
                     → escalation safety check → dataset growth)

    After completion, launch the dashboard with option [5] to explore.
    """
    print("\n" + "🔁" * 30)
    print("RUNNING FULL PIPELINE (Stages 1 → 2 → 3 → 4)")
    print("🔁" * 30)

    state_start = _snapshot_state()

    if not stage_1_scrape_and_preprocess():
        print("\n⚠ Stage 1 failed. Pipeline halted.")
        return
    if not stage_2_train_relevance():
        print("\n⚠ Stage 2 failed. Pipeline halted.")
        return
    if not stage_3_train_classifiers():
        print("\n⚠ Stage 3 failed. Pipeline halted.")
        return
    stage_4_run()

    print("\n" + "=" * 60)
    print("✅ Full pipeline complete!")
    print("=" * 60)
    _print_pipeline_summary(
        state_start, _snapshot_state(),
        title="FULL PIPELINE SUMMARY",
    )
    print("   Launch dashboard with option [5] to explore results.")
    print("=" * 60)


# ===========================================================================
# FULL SYSTEM CHECK
# ===========================================================================

def run_system_check():
    """
    Verify that all components are present and functional.

    Checks:
        1. Data files exist
        2. Pipeline scripts exist
        3. Trained models exist
        4. Module imports succeed
        5. Database health
    """
    print("\n" + "=" * 60)
    print("RNIA — Full System Check")
    print("=" * 60)

    all_ok = True

    # --- 1. Data files ---
    print("\n[1/5] Checking data files...")
    relevance_csv = os.path.join(
        PROJECT_ROOT, "data", "labeled_dataset", "relevance_dataset.csv"
    )
    data_files = [
        ("Raw news dataset",       RAW_NEWS_CSV),
        ("Cleaned dataset",        CLEAN_NEWS_CSV),
        ("Gold (rel=1) dataset",   LABELED_CSV),
        ("Relevance dataset",      relevance_csv),
        ("Analysis report",        REPORT_CSV),
    ]
    for name, path in data_files:
        if os.path.isfile(path):
            import pandas as pd
            try:
                full_df = pd.read_csv(path, encoding="utf-8-sig")
                print(f"  ✅  {name}: {len(full_df)} rows")
            except Exception:
                print(f"  ✅  {name}: exists")
        else:
            print(f"  ⚠   {name}: NOT FOUND")

    # --- 2. Pipeline scripts ---
    print("\n[2/5] Checking pipeline scripts...")
    scripts = [
        ("scrapers/news_scraper.py",           os.path.join(PROJECT_ROOT, "scrapers", "news_scraper.py")),
        ("preprocessing/clean_text.py",        os.path.join(PROJECT_ROOT, "preprocessing", "clean_text.py")),
        ("models/train_models.py",             os.path.join(PROJECT_ROOT, "models", "train_models.py")),
        ("pipeline/ingest.py",                 os.path.join(PROJECT_ROOT, "pipeline", "ingest.py")),
        ("pipeline/run.py",                    os.path.join(PROJECT_ROOT, "pipeline", "run.py")),
        ("pipeline/keywords.yaml",             os.path.join(PROJECT_ROOT, "pipeline", "keywords.yaml")),
        ("scripts/export_gold_dataset.py",     os.path.join(PROJECT_ROOT, "scripts", "export_gold_dataset.py")),
        ("scripts/export_relevance_dataset.py", os.path.join(PROJECT_ROOT, "scripts", "export_relevance_dataset.py")),
        ("evaluation/evaluate_models.py",      os.path.join(PROJECT_ROOT, "evaluation", "evaluate_models.py")),
        ("backend/app.py",                     os.path.join(PROJECT_ROOT, "backend", "app.py")),
        ("backend/queries.py",                 os.path.join(PROJECT_ROOT, "backend", "queries.py")),
    ]
    for name, path in scripts:
        exists = os.path.isfile(path)
        status = "✅" if exists else "⚠ MISSING"
        print(f"  {status}  {name}")
        if not exists:
            all_ok = False

    # --- 3. Trained models ---
    print("\n[3/5] Checking trained models...")
    model_files = [
        # Required at runtime
        "relevance_classifier.pkl",
        "tfidf_relevance_vectorizer.pkl",
        "event_classifier.pkl",
        "tfidf_vectorizer.pkl",
        "stance_detector.pkl",
        "tfidf_stance_vectorizer.pkl",
        "tfidf_subscore_vectorizer.pkl",
        "subscore_materiality.pkl",
        "subscore_market_linkage.pkl",
        "subscore_time_sensitivity.pkl",
        "subscore_credibility.pkl",
        # Legacy single-impact regressor (unused at runtime but kept on disk)
        "impact_regressor.pkl",
        "tfidf_impact_vectorizer.pkl",
    ]
    runtime_required = {
        "relevance_classifier.pkl", "event_classifier.pkl",
        "stance_detector.pkl", "tfidf_subscore_vectorizer.pkl",
    }
    for mf in model_files:
        path = os.path.join(SAVED_MODELS_DIR, mf)
        exists = os.path.isfile(path)
        status = "✅" if exists else "⚠ MISSING"
        print(f"  {status}  {mf}")
        if not exists and mf in runtime_required:
            all_ok = False

    # --- 4. Module imports ---
    print("\n[4/5] Checking module imports...")
    modules_to_check = [
        ("models.relevance_classifier", "RelevanceClassifierLR"),
        ("models.sub_score_regressor",  "SubScoreRegressor"),
        ("models.event_classifier",     "EventClassifierLR"),
        ("models.stance_detector",      "StanceDetectorLR"),
        ("pipeline.ingest",             "ingest_clean_csv"),
        ("pipeline.run",                "run"),
        ("backend.queries",             "summary_stats"),
        ("backend.app",                 "app"),
    ]
    for module_name, attr_name in modules_to_check:
        try:
            mod = __import__(module_name, fromlist=[attr_name])
            getattr(mod, attr_name)
            print(f"  ✅  {module_name}.{attr_name}")
        except Exception as exc:
            print(f"  ❌  {module_name}.{attr_name} — {exc}")
            all_ok = False

    # --- 5. Database ---
    print("\n[5/5] Checking database...")
    if os.path.isfile(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            expected = ["articles", "v2_analyses"]
            for t in expected:
                if t in tables:
                    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    print(f"  ✅  Table '{t}': {count} rows")
                else:
                    print(f"  ⚠   Table '{t}': MISSING")
                    all_ok = False

            # Quick analysis stats
            try:
                done = conn.execute(
                    "SELECT COUNT(*) FROM v2_analyses WHERE processing_status='done' AND relevance=1"
                ).fetchone()[0]
                print(f"  ✅  Relevant & analysed: {done}")
                v3_done = conn.execute(
                    "SELECT COUNT(*) FROM v2_analyses WHERE model_used='v3-ml'"
                ).fetchone()[0]
                v3_filt = conn.execute(
                    "SELECT COUNT(*) FROM v2_analyses WHERE model_used='v3-keyword-filter'"
                ).fetchone()[0]
                v3_esc = conn.execute(
                    "SELECT COUNT(*) FROM v2_analyses WHERE model_used='v3-ml' AND processing_status='needs_escalation'"
                ).fetchone()[0]
                print(f"  ✅  V3 ML labelled       : {v3_done}  (escalated: {v3_esc})")
                print(f"  ✅  V3 keyword-filtered  : {v3_filt}")
            except Exception:
                pass

            conn.close()
        except Exception as exc:
            print(f"  ❌  Database error: {exc}")
            all_ok = False
    else:
        print(f"  ⚠   Database not found: {DB_PATH}")

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ ALL CHECKS PASSED — System is ready!")
    else:
        print("⚠ SOME CHECKS FAILED — Review issues above.")
    print("=" * 60)

    return all_ok


# ===========================================================================
# MENU
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
    print("  [1]  Scrape + preprocess + ingest + classify new articles")
    print("  [2]  Train relevance classifier (full ~5,400-row dataset)")
    print("  [3]  Train classifier suite (event + stance + sub-scores)")
    print("  [4]  Run pipeline on new articles")
    print("  [5]  Launch web dashboard (FastAPI + React)")
    print("  [6]  Run model evaluation")
    print("  [9]  Run FinBERT comparison baseline (stance only)")
    print("  ─────────────────────────────────────────")
    print("  [7]  Run full pipeline (Stages 1→2→3→4)")
    print("  [8]  Run full system check")
    print("  [0]  Exit")
    print()

    choice = input("  Enter your choice [0-9]: ").strip()
    return choice


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":
    # Menu-driven pipeline controller
    while True:
        choice = show_menu()

        if choice == "1":
            stage_1_scrape_and_preprocess()
        elif choice == "2":
            stage_2_train_relevance()
        elif choice == "3":
            stage_3_train_classifiers()
        elif choice == "4":
            stage_4_run()
        elif choice == "5":
            stage_7_dashboard()
        elif choice == "6":
            stage_6_evaluation()
        elif choice == "7":
            run_full_pipeline()
        elif choice == "8":
            run_system_check()
        elif choice == "9":
            stage_9_finbert_comparison()
        elif choice == "0":
            print("\nGoodbye! 👋")
            sys.exit(0)
        else:
            print("\n⚠ Invalid choice. Please enter a number from 0 to 9.")

        input("\nPress Enter to return to the menu...")

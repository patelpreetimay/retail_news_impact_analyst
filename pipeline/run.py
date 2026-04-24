"""
pipeline/run.py — Unified Pipeline Driver
==========================================

Runs the full ML pipeline on a set of articles in data/rnia.db:

    Step 1  Deterministic keyword filter (loaded from pipeline/keywords.yaml)
    Step 2  ML relevance classifier (models/relevance_classifier.py)
    Step 3  Event classifier + stance detector + sub-score regressors
    Step 4  Deterministic impact_score formula (sub_score_regressor.compute_impact_score)
    Step 5  Escalation safety check (per docs/pipeline-algorithm-spec.md § 4)
    Step 6  Write v2_analyses row (status = done | needs_escalation)
    Step 7  Grow datasets:
              - relevance_dataset.csv ← append ALL processed articles
              - financial_news_labeled.csv ← append only high-confidence rel=1 rows
                (confidence gate, option C from the implementation plan)

This script does NOT scrape or call Gemini. The Gemini API was used once
to label the seed dataset; everything in this driver is pure ML + Python.

Usage:
    # Process every article that has no v2_analyses row yet
    python -m pipeline.run

    # Or programmatically with explicit ids
    >>> from pipeline.run import run
    >>> run(article_ids=[5467, 5468, 5469])
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.event_classifier import EventClassifierLR
from models.stance_detector import StanceDetectorLR
from models.relevance_classifier import RelevanceClassifierLR
from models.sub_score_regressor import (
    SUB_SCORES,
    SubScoreRegressor,
    compute_impact_score,
)
from utils.text_features import build_input_text, get_headline_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(PROJECT_ROOT, "data", "rnia.db")
KEYWORDS_YML = os.path.join(PROJECT_ROOT, "pipeline", "keywords.yaml")
RELEVANCE_DATASET_CSV = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "relevance_dataset.csv"
)
GOLD_DATASET_CSV = os.path.join(
    PROJECT_ROOT, "data", "labeled_dataset", "financial_news_labeled.csv"
)

# ---------------------------------------------------------------------------
# Config — knobs documented in docs/pipeline-algorithm-spec.md
# ---------------------------------------------------------------------------
MIN_BODY_LENGTH = 300                  # § 1 rule 2
JUNK_STUB_THRESHOLD = 600              # ingest-side
LOW_CONFIDENCE_THRESHOLD = 0.6         # § 4 trigger
HIGH_CONFIDENCE_GATE = 0.85            # option C — feedback into gold dataset
INCONSISTENT_HIGH = 0.85
INCONSISTENT_LOW = 0.20
HIGH_STAKES_EVENTS = {"Regulatory_Action", "Legal_Action"}

# Mapping ML stance (lowercase) → DB stance (uppercase) for v2_analyses
STANCE_ML_TO_DB = {
    "bullish": "BULLISH",
    "bearish": "BEARISH",
    "neutral": "NEUTRAL",
    "mixed":   "MIXED",
}

# The taxonomy collapses 9 legacy event names → 7. The v2_analyses table
# CHECK constraint was created for the legacy 9 names, so any write of
# "Market_Movement" (the collapsed bucket) violates the constraint. Map back
# to a legacy name on DB writes only — gold CSV + frontend logic still see
# the canonical name.
EVENT_TO_DB_LEGACY = {
    "Market_Movement": "Other",
}


def _event_for_db(event: str) -> str:
    """Translate a canonical event label to a legacy DB-allowed label."""
    return EVENT_TO_DB_LEGACY.get(event, event)


# ---------------------------------------------------------------------------
# Stage-1 keyword filter (deterministic) — keeps the same logic the
# legacy v2 stage-1 module used so re-runs are bit-identical.
# ---------------------------------------------------------------------------

def _load_keyword_pattern(path: str = KEYWORDS_YML) -> re.Pattern:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    terms: set[str] = set()
    for cat, words in raw.items():
        for w in words:
            if w and str(w).strip():
                terms.add(str(w).strip().lower())
    parts = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def _keyword_filter(
    headline: str,
    clean_text: str,
    is_junk_stub: int,
    body_length: int,
    pattern: re.Pattern,
) -> tuple[int, str]:
    """Return (relevance, reason). Mirrors the legacy stage-1 decide() rule."""
    if is_junk_stub == 1:
        return 0, "junk_stub"
    if body_length < MIN_BODY_LENGTH:
        return 0, f"body_too_short (<{MIN_BODY_LENGTH} chars)"
    haystack = f"{headline or ''} {clean_text or ''}"
    if pattern.search(haystack):
        return 1, "matched_keywords"
    return 0, "no_keyword_match"


# ---------------------------------------------------------------------------
# Escalation safety check — § 4 of the spec
# ---------------------------------------------------------------------------

def _escalation_reason(
    event_conf: float,
    stance_conf: float,
    sub_scores: dict[str, float],
    event_type: str,
    stance_db: str,
) -> str | None:
    if event_conf < LOW_CONFIDENCE_THRESHOLD:
        return f"low_event_confidence={event_conf:.2f}"
    if stance_conf < LOW_CONFIDENCE_THRESHOLD:
        return f"low_stance_confidence={stance_conf:.2f}"
    mat = sub_scores["materiality"]
    lnk = sub_scores["market_linkage"]
    if mat >= INCONSISTENT_HIGH and lnk <= INCONSISTENT_LOW:
        return "inconsistent_high_mat_low_link"
    if mat <= INCONSISTENT_LOW and lnk >= INCONSISTENT_HIGH:
        return "inconsistent_low_mat_high_link"
    if stance_db == "BEARISH" and event_type in HIGH_STAKES_EVENTS:
        return "high_stakes_bearish_regulatory_or_legal"
    return None


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _select_articles(
    conn: sqlite3.Connection,
    article_ids: list[int] | None,
) -> pd.DataFrame:
    """
    Pull article rows that need processing. If article_ids is None, pull
    every article that has no v2_analyses row yet.
    """
    if article_ids is None:
        sql = """
            SELECT a.article_id, a.headline, a.clean_text,
                   a.is_junk_stub, a.body_length
            FROM articles a
            LEFT JOIN v2_analyses v ON v.article_id = a.article_id
            WHERE v.article_id IS NULL
            ORDER BY a.article_id
        """
        return pd.read_sql_query(sql, conn)

    if not article_ids:
        return pd.DataFrame(
            columns=["article_id", "headline", "clean_text",
                     "is_junk_stub", "body_length"]
        )

    placeholders = ",".join("?" * len(article_ids))
    sql = f"""
        SELECT article_id, headline, clean_text,
               is_junk_stub, body_length
        FROM articles
        WHERE article_id IN ({placeholders})
        ORDER BY article_id
    """
    return pd.read_sql_query(sql, conn, params=article_ids)


_UPSERT_FILTERED = """
INSERT INTO v2_analyses (
    article_id, processing_status, filter_reason,
    relevance, model_used,
    created_at, updated_at
) VALUES (?, 'filtered', ?, 0, ?, ?, ?)
ON CONFLICT(article_id) DO UPDATE SET
    processing_status='filtered',
    filter_reason=excluded.filter_reason,
    relevance=0,
    model_used=excluded.model_used,
    updated_at=excluded.updated_at
"""

_UPSERT_DONE = """
INSERT INTO v2_analyses (
    article_id, processing_status,
    relevance, event_type, stance,
    materiality, market_linkage, time_sensitivity, credibility, impact_score,
    reasoning, model_used,
    gemini_confidence_event, gemini_confidence_stance,
    escalation_reason,
    created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(article_id) DO UPDATE SET
    processing_status=excluded.processing_status,
    relevance=excluded.relevance,
    event_type=excluded.event_type,
    stance=excluded.stance,
    materiality=excluded.materiality,
    market_linkage=excluded.market_linkage,
    time_sensitivity=excluded.time_sensitivity,
    credibility=excluded.credibility,
    impact_score=excluded.impact_score,
    reasoning=excluded.reasoning,
    model_used=excluded.model_used,
    gemini_confidence_event=excluded.gemini_confidence_event,
    gemini_confidence_stance=excluded.gemini_confidence_stance,
    escalation_reason=excluded.escalation_reason,
    updated_at=excluded.updated_at
"""


# ---------------------------------------------------------------------------
# Dataset growth helpers (continuous learning)
# ---------------------------------------------------------------------------

def _append_to_relevance_csv(rows: list[dict]) -> None:
    """Grow the relevance training set with newly-processed articles."""
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.isfile(RELEVANCE_DATASET_CSV):
        existing = pd.read_csv(RELEVANCE_DATASET_CSV, encoding="utf-8-sig")
        df = pd.concat([existing, df_new], ignore_index=True)
        # Dedupe by article_id, keeping the latest decision
        df = df.drop_duplicates(subset=["article_id"], keep="last")
    else:
        df = df_new
    os.makedirs(os.path.dirname(RELEVANCE_DATASET_CSV), exist_ok=True)
    df.to_csv(RELEVANCE_DATASET_CSV, index=False, encoding="utf-8-sig")


def _append_to_gold_csv(rows: list[dict]) -> None:
    """
    Grow the gold (event/stance/impact) training set — option C:
    only high-confidence relevance=1 predictions are appended.
    """
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.isfile(GOLD_DATASET_CSV):
        existing = pd.read_csv(GOLD_DATASET_CSV, encoding="utf-8-sig")
        # Make sure column sets align
        for col in df_new.columns:
            if col not in existing.columns:
                existing[col] = None
        for col in existing.columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[existing.columns]
        df = pd.concat([existing, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["url"], keep="last")
    else:
        df = df_new
    os.makedirs(os.path.dirname(GOLD_DATASET_CSV), exist_ok=True)
    df.to_csv(GOLD_DATASET_CSV, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(
    article_ids: list[int] | None = None,
    db_path: str = DB_PATH,
    grow_datasets: bool = True,
    high_confidence_gate: float = HIGH_CONFIDENCE_GATE,
) -> dict:
    """
    Run the full ML pipeline on the given article_ids.

    Parameters
    ----------
    article_ids : list[int] | None
        Specific article ids to process. If None, processes every article
        in the DB that has no v2_analyses row yet.
    db_path : str
        Path to data/rnia.db.
    grow_datasets : bool
        If True, append processed rows to the two training CSVs.
    high_confidence_gate : float
        Min(event_conf, stance_conf) required to feed a rel=1 prediction
        back into the gold dataset (option C).

    Returns
    -------
    dict
        Summary counters.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    # ---- load models ------------------------------------------------------
    logger.info("loading ML models...")
    rel_clf = RelevanceClassifierLR();   rel_clf.load_model()
    event_clf = EventClassifierLR();     event_clf.load_model()
    stance_det = StanceDetectorLR();     stance_det.load_model()
    sub_reg = SubScoreRegressor();       sub_reg.load_model()
    pattern = _load_keyword_pattern()

    # ---- pull rows --------------------------------------------------------
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    rows = _select_articles(conn, article_ids)
    n = len(rows)
    logger.info("articles to process: %d", n)
    if n == 0:
        conn.close()
        return {"processed": 0}

    counters = {
        "filtered_keyword": 0,
        "ml_irrelevant":    0,
        "ml_relevant":      0,
        "needs_escalation": 0,
        "high_conf_to_gold": 0,
    }
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    relevance_csv_rows: list[dict] = []
    gold_csv_rows: list[dict] = []

    cur = conn.cursor()
    for _, r in rows.iterrows():
        aid       = int(r["article_id"])
        headline  = str(r["headline"] or "")
        clean     = str(r["clean_text"] or "")
        body_len  = int(r["body_length"] or 0)
        junk_stub = int(r["is_junk_stub"] or 0)

        # Step 1 — deterministic keyword filter
        kf_rel, kf_reason = _keyword_filter(
            headline, clean, junk_stub, body_len, pattern,
        )

        if kf_rel == 0:
            cur.execute(
                _UPSERT_FILTERED,
                (aid, kf_reason, "v3-keyword-filter", now_iso, now_iso),
            )
            counters["filtered_keyword"] += 1
            relevance_csv_rows.append({
                "article_id": aid,
                "headline": headline, "clean_text": clean,
                "source": None, "region": None, "timestamp": None, "url": None,
                "relevance": 0, "relevance_source": "v3_keyword_filter",
            })
            continue

        # Step 2 — ML relevance classifier (uses shared text builder so
        # headline-weighting at training time is mirrored at inference time)
        text = build_input_text(headline, clean)
        p_rel = rel_clf.predict_proba(text)
        ml_rel = int(p_rel >= 0.5)

        if ml_rel == 0:
            cur.execute(
                _UPSERT_DONE,
                (aid, "done", 0, None, None, None, None, None, None, None,
                 f"ML relevance model rejected (p={p_rel:.3f}).",
                 "v3-ml", None, None, None, now_iso, now_iso),
            )
            counters["ml_irrelevant"] += 1
            relevance_csv_rows.append({
                "article_id": aid,
                "headline": headline, "clean_text": clean,
                "source": None, "region": None, "timestamp": None, "url": None,
                "relevance": 0, "relevance_source": "v3_ml_model",
            })
            continue

        # Step 3 — event / stance / sub-scores
        event = event_clf.predict_event(text)
        event_probs = event_clf.predict_proba(text)
        event_conf = max(event_probs.values()) if event_probs else 0.0

        stance_lc = stance_det.predict_stance(text)
        stance_db = STANCE_ML_TO_DB.get(stance_lc.lower(), "NEUTRAL")
        stance_probs_obj = getattr(stance_det, "predict_proba", None)
        if callable(stance_probs_obj):
            try:
                stance_probs = stance_probs_obj(text)
                stance_conf = max(stance_probs.values()) if stance_probs else 0.0
            except Exception:
                stance_conf = 0.0
        else:
            stance_conf = 0.0

        sub_pred = sub_reg.predict_subscores(text)
        impact = compute_impact_score(sub_pred)

        # Step 5 — escalation
        esc_reason = _escalation_reason(
            event_conf, stance_conf, sub_pred, event, stance_db,
        )
        status = "needs_escalation" if esc_reason else "done"
        if esc_reason:
            counters["needs_escalation"] += 1
        counters["ml_relevant"] += 1

        reasoning = (
            f"ML pipeline: event={event} (conf={event_conf:.2f}), "
            f"stance={stance_db} (conf={stance_conf:.2f}), "
            f"impact={impact:.3f}."
        )
        if esc_reason:
            reasoning += f" Flagged: {esc_reason}."

        cur.execute(
            _UPSERT_DONE,
            (
                aid, status,
                1, _event_for_db(event), stance_db,
                sub_pred["materiality"], sub_pred["market_linkage"],
                sub_pred["time_sensitivity"], sub_pred["credibility"],
                impact,
                reasoning, "v3-ml",
                event_conf, stance_conf,
                esc_reason,
                now_iso, now_iso,
            ),
        )

        relevance_csv_rows.append({
            "article_id": aid,
            "headline": headline, "clean_text": clean,
            "source": None, "region": None, "timestamp": None, "url": None,
            "relevance": 1, "relevance_source": "v3_ml_model",
        })

        # Confidence-gated feedback into the gold dataset (option C)
        if (
            grow_datasets
            and esc_reason is None
            and min(event_conf, stance_conf) >= high_confidence_gate
        ):
            gold_csv_rows.append({
                "headline": headline,
                "clean_text": clean,
                "event_type": event,
                "stance": stance_lc.lower(),
                "impact_score": impact,
                "materiality":      sub_pred["materiality"],
                "market_linkage":   sub_pred["market_linkage"],
                "time_sensitivity": sub_pred["time_sensitivity"],
                "credibility":      sub_pred["credibility"],
                "source": None, "timestamp": None,
                "url": f"v3://aid/{aid}",  # placeholder; URL needed for dedup
                "region": None, "relevance": "RELEVANT",
            })
            counters["high_conf_to_gold"] += 1

    conn.commit()
    conn.close()

    # ---- grow training datasets ------------------------------------------
    if grow_datasets:
        _append_to_relevance_csv(relevance_csv_rows)
        if gold_csv_rows:
            _append_to_gold_csv(gold_csv_rows)

    counters["processed"] = n
    return counters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    summary = run()
    print()
    print("=" * 60)
    print("PIPELINE — SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print("=" * 60)

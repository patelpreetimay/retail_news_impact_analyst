"""
backend/app.py — RNIA FastAPI backend
======================================

Serves the pipeline output (rnia.db.v2_analyses) to the React/Vite
frontend at frontend/. Read-only — never writes the DB.

Endpoints
---------
    GET  /         — health check
    GET  /news     — list of analysed articles (frontend-shaped records)
    GET  /summary  — dashboard summary metrics
    POST /analyze  — ad-hoc article analysis using the trained ML models

Run
---
    uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable for top-level packages.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import queries as Q  # noqa: E402
from scoring.impact_score import calculate_impact_from_components, calculate_recency  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rnia.backend")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RNIA API",
    version="1.0.0",
    description="Retail News Impact Analyst — ML pipeline analyses served from rnia.db.v2_analyses",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Vocabulary mappers — v2 (DB) ↔ frontend (data.js)
# ---------------------------------------------------------------------------
# Import from canonical taxonomy to stay in sync
from taxonomy.event_taxonomy import V2_TO_FRONTEND_EVENT

V2_TO_FE_EVENT = V2_TO_FRONTEND_EVENT

# Stance mapper — handles both DB format (UPPERCASE) and ML format (lowercase)
V2_TO_FE_STANCE = {
    "BULLISH":      "bullish",
    "BEARISH":      "bearish",
    "NEUTRAL":      "neutral",
    "MIXED":        "neutral",
    "UNCLASSIFIED": "neutral",
    # ML pipeline outputs lowercase
    "bullish":      "bullish",
    "bearish":      "bearish",
    "neutral":      "neutral",
    "mixed":        "neutral",
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return float(f"{f:.4f}")


def _best_timestamp(row: dict) -> str:
    for key in ("published_at", "scraped_at", "timestamp"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in ("nan", "nat", "none"):
            return text
    return ""


def _live_score_parts(row: dict) -> tuple[float, float, float, float]:
    """
    Return (impact_score, recency, materiality, credibility) for live display.

    The DB value remains an audit-time snapshot. This function recomputes the
    recency-dependent score on every API request so refreshes do not show stale
    impact values for older articles.
    """
    timestamp = _best_timestamp(row)
    materiality = _safe_float(row.get("materiality"), 0.5)
    credibility = _safe_float(row.get("credibility"), 0.7)
    recency = calculate_recency(timestamp)
    impact_score = calculate_impact_from_components(
        materiality=materiality,
        credibility=credibility,
        recency=recency,
    )
    return impact_score, recency, materiality, credibility


def _row_to_frontend(row: dict) -> dict:
    """Shape one v2_analyses row into what the frontend expects."""
    timestamp = _best_timestamp(row)
    impact_score, recency, materiality, credibility = _live_score_parts(row)
    return {
        "id":           row.get("article_id"),
        "headline":     row.get("headline") or "",
        "event_type":   V2_TO_FE_EVENT.get(row.get("event_type") or "", "market"),
        "stance":       V2_TO_FE_STANCE.get(row.get("stance") or "", "neutral"),
        "impact_score": impact_score,
        "explanation":  row.get("reasoning") or "",
        "source":       row.get("source") or "unknown",
        # Fall back to scraped_at when the RSS feed didn't provide a pubDate,
        # so the UI never shows an empty timestamp.
        "timestamp":    timestamp,
        # Extras the frontend can opportunistically use
        "url":          row.get("url") or "",
        "materiality":     materiality,
        "market_linkage":  _safe_float(row.get("market_linkage")),
        "time_sensitivity":_safe_float(row.get("time_sensitivity")),
        "credibility":     credibility,
        "recency":         recency,
        "stored_impact_score": _safe_float(row.get("impact_score")),
        "model_used":      row.get("model_used") or "",
    }


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/")
def health_check():
    try:
        s = Q.summary_stats()
        return {
            "status":           "RNIA API is running",
            "db_path":          str(Q.DB_PATH),
            "total_articles":   int(s["overall"].iloc[0]["total_articles"]),
            "analysed":         int(s["overall"].iloc[0]["analysed"]),
            "relevant":         int(s["overall"].iloc[0]["relevant"]),
        }
    except Exception as exc:  # pragma: no cover
        logger.error("health check failed: %s", exc)
        return {"status": "RNIA API is running", "db_error": str(exc)}


@app.get("/news")
def get_news(limit: int = 2500):
    """Return up to *limit* relevant analysed articles, newest first."""
    try:
        df = Q.get_latest_articles(limit=limit)
    except Exception as exc:
        logger.exception("/news failed")
        raise HTTPException(status_code=500, detail=str(exc))

    records = df.to_dict(orient="records")
    out = [_row_to_frontend(r) for r in records]
    logger.info("GET /news → %d articles", len(out))
    return out


@app.get("/summary")
def get_summary():
    """Aggregate counts shaped for the Overview page."""
    try:
        s = Q.summary_stats()
    except Exception as exc:
        logger.exception("/summary failed")
        raise HTTPException(status_code=500, detail=str(exc))

    overall = s["overall"].iloc[0]
    total_articles = int(overall["total_articles"])
    relevant       = int(overall["relevant"])

    # Average live impact across the relevant + done set. Stored DB scores are
    # audit snapshots; live scores include recency decay as of this request.
    df_top = Q.top_articles_by_impact(n=10_000)
    if len(df_top):
        live_scores = [
            _live_score_parts(r)[0]
            for r in df_top.to_dict(orient="records")
        ]
        avg_impact = _safe_float(sum(live_scores) / len(live_scores))
    else:
        avg_impact = 0.0

    # Event distribution → frontend ids
    event_distribution: dict[str, int] = {}
    for _, r in s["event_type"].iterrows():
        fe_id = V2_TO_FE_EVENT.get(r["event_type"], "market")
        event_distribution[fe_id] = event_distribution.get(fe_id, 0) + int(r["n"])

    # Stance distribution → frontend ids
    stance_distribution: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
    for _, r in s["stance"].iterrows():
        fe_id = V2_TO_FE_STANCE.get(r["stance"], "neutral")
        stance_distribution[fe_id] = stance_distribution.get(fe_id, 0) + int(r["n"])

    # Top event by count
    top_event = "Unknown"
    if len(s["event_type"]):
        top_v2 = s["event_type"].iloc[0]["event_type"]
        top_event = top_v2.replace("_", " ").title()

    payload = {
        "total_articles":       relevant,         # what the user sees on cards
        "raw_total_articles":   total_articles,   # full corpus, for diagnostics
        "average_impact_score": avg_impact,
        "event_distribution":   event_distribution,
        "stance_distribution":  stance_distribution,
        "top_event":            top_event,
    }
    logger.info("GET /summary → relevant=%d, avg_impact=%.3f, top=%s",
                relevant, avg_impact, top_event)
    return payload


# ---------------------------------------------------------------------------
# /analyze — ML-powered ad-hoc article analysis
# Uses the same trained V3 models (EventClassifierLR, StanceDetectorLR,
# SubScoreRegressor) that power the News Feed pipeline, ensuring
# consistent classification across the entire application.
# ---------------------------------------------------------------------------

from models.event_classifier import EventClassifierLR    # noqa: E402
from models.stance_detector import StanceDetectorLR      # noqa: E402
from models.sub_score_regressor import (                 # noqa: E402
    SubScoreRegressor,
)
from models.keyword_rules import (                       # noqa: E402
    KeywordEventClassifier,
    KeywordStanceClassifier,
)
from models.ensemble import load_ensemble_config         # noqa: E402

# Load models once at startup — same models the V3 pipeline uses
logger.info("Loading V3 ML models for /analyze endpoint …")
try:
    _event_clf = EventClassifierLR();  _event_clf.load_model()
    _stance_det = StanceDetectorLR();  _stance_det.load_model()
    _sub_reg = SubScoreRegressor();    _sub_reg.load_model()
    _kw_event = KeywordEventClassifier()
    _kw_stance = KeywordStanceClassifier()
    _ens_cfg = load_ensemble_config()
    _BASE_EVENT_KW_WEIGHT  = float(_ens_cfg.get("event_kw_weight", 0.10))
    _BASE_STANCE_KW_WEIGHT = float(_ens_cfg.get("stance_kw_weight", 0.05))
    _models_loaded = True
    logger.info(
        "V3 ML models + keyword ensemble loaded (base kw_weight: event=%.2f stance=%.2f).",
        _BASE_EVENT_KW_WEIGHT, _BASE_STANCE_KW_WEIGHT,
    )
except Exception as _load_err:
    logger.error("Failed to load V3 ML models: %s — /analyze will be unavailable.", _load_err)
    _models_loaded = False


# ---------------------------------------------------------------------------
# Ensemble helpers — combine ML probabilities with keyword-rule signal.
# Short headlines have too few TF-IDF features for the LR models to lock on;
# the curated keyword classifier reads the same financial vocabulary humans
# do (crash, plunge, fraud, beat, surge…) and rescues those cases.
# Length-aware weighting: keyword weight is high on short text and falls to
# the calibrated full-article default once the input is article-sized.
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _length_aware_kw_weight(text: str, base: float, short: float) -> float:
    """Interpolate keyword weight from `short` (≤15 words) to `base` (≥80 words)."""
    n = _word_count(text)
    if n >= 80:
        return base
    if n <= 15:
        return max(base, short)
    frac = (80 - n) / (80 - 15)
    return base + (short - base) * frac


_HEADLINE_PREFIX_RE = re.compile(
    r"^\s*(BT\s+EXPLAINER|EXPLAINED?|EXPLAINER|EXCLUSIVE|BREAKING|UPDATE\s*\d*|"
    r"LIVE(?:\s+UPDATES?)?|WATCH|VIDEO|PHOTOS?|OPINION|ANALYSIS|EDITORIAL|"
    r"FLASH|REPORT|NEWS|MARKETS?)"
    r"\s*[\|:\-–—]+\s*",
    re.IGNORECASE,
)
_TICKER_PREFIX_RE = re.compile(r"^\s*[\[\(][^\]\)]+[\]\)]\s*")
_DATE_PREFIX_RE = re.compile(
    r"^\s*(?:[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4}|"     # "Mar 12, 2024"
    r"\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4}|"              # "12 Mar 2024"
    r"\d{4}-\d{2}-\d{2})"                              # "2024-03-12"
    r"\s*[\|\-–—:]+\s*",
)
_TRAILING_ATTRIBUTION_RE = re.compile(
    r"\s*[—–\-]\s*(?:according to|per|via|reports?|said|writes?)\s+"
    r"[A-Z][\w\s,\.&]+?(?:\.|$)",
    re.IGNORECASE,
)
_PHOTO_CAPTION_RE = re.compile(
    r"\(\s*(?:photo|image|file photo|reuters|getty|ap photo)[^)]*\)",
    re.IGNORECASE,
)


def _clean_input_text(text: str) -> str:
    """Strip newsroom-only prefixes that aren't represented in the training data."""
    cleaned = (text or "").strip()
    # Drop photo captions and trailing attributions anywhere in the text.
    cleaned = _PHOTO_CAPTION_RE.sub(" ", cleaned)
    cleaned = _TRAILING_ATTRIBUTION_RE.sub("", cleaned)
    # Strip up to 3 stacked leading prefixes (e.g. "BREAKING | EXCLUSIVE | (Reuters) | …")
    for _ in range(3):
        new = _HEADLINE_PREFIX_RE.sub("", cleaned)
        new = _TICKER_PREFIX_RE.sub("", new)
        new = _DATE_PREFIX_RE.sub("", new)
        if new == cleaned:
            break
        cleaned = new
    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# Stored-article lookup — pasting a known headline pulls in the full body
# so the analyser produces the same answer the News Feed already shows.
# ---------------------------------------------------------------------------
import sqlite3                                                       # noqa: E402
from pathlib import Path                                             # noqa: E402

_QUOTE_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _norm_for_match(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.translate(_QUOTE_FOLD).strip().lower())


def _lookup_stored_article(text: str) -> dict | None:
    """If `text` matches a stored article's headline, return its full record.

    Match rules (in order):
      1. Exact normalised equality.
      2. Pasted text starts with a stored headline (user pasted headline + extra).
      3. Stored headline starts with the pasted text (user pasted a prefix).
    Returns None if nothing meets the bar.
    """
    target = _norm_for_match(text)
    if len(target) < 20:
        return None
    db_path = Path(Q.DB_PATH)
    if not db_path.exists():
        return None
    try:
        uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT article_id, headline, clean_text "
            "FROM articles WHERE headline IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception as exc:                                         # pragma: no cover
        logger.warning("article lookup failed: %s", exc)
        return None
    for r in rows:
        h = _norm_for_match(r["headline"])
        if not h or len(h) < 20:
            continue
        if h == target or target.startswith(h) or h.startswith(target):
            return {
                "article_id": r["article_id"],
                "headline":   r["headline"],
                "clean_text": r["clean_text"] or "",
            }
    return None


def _ensemble_event(
    text: str,
) -> tuple[str, dict[str, float], float, dict[str, list[str]]]:
    """Return (predicted_class, combined_probs, kw_weight_used, matched_terms)."""
    lr_probs = _event_clf.predict_proba(text)
    kw_scores, kw_evidence = _kw_event.predict_with_evidence(text)
    kw_w = _length_aware_kw_weight(text, _BASE_EVENT_KW_WEIGHT, short=0.35)
    lr_w = 1.0 - kw_w
    classes = set(lr_probs.keys()) | set(kw_scores.keys())
    combined = {
        c: lr_w * lr_probs.get(c, 0.0) + kw_w * kw_scores.get(c, 0.0)
        for c in classes
    }
    return max(combined, key=combined.get), combined, kw_w, kw_evidence


_STANCE_NEUTRAL_THRESHOLD = 0.45


def _ensemble_stance(
    text: str,
) -> tuple[str, dict[str, float], float, dict[str, list[str]]]:
    """Return (predicted_class, combined_probs, kw_weight_used, matched_terms).

    When neither signal source is confident (combined max < threshold AND no
    keyword hits), defer to "neutral" rather than emit a low-confidence
    bullish/bearish call — same defensive rule the LR model uses in strict
    mode at [stance_detector.py:180].
    """
    lr_probs = _stance_det.predict_proba(text)
    kw_scores, kw_evidence = _kw_stance.predict_with_evidence(text)
    kw_w = _length_aware_kw_weight(text, _BASE_STANCE_KW_WEIGHT, short=0.55)
    lr_w = 1.0 - kw_w
    classes = set(lr_probs.keys()) | set(kw_scores.keys())
    combined = {
        c: lr_w * lr_probs.get(c, 0.0) + kw_w * kw_scores.get(c, 0.0)
        for c in classes
    }
    best = max(combined, key=combined.get)
    kw_total = sum(kw_scores.values())
    if combined[best] < _STANCE_NEUTRAL_THRESHOLD and kw_total == 0:
        # Genuine "no signal" case — be honest rather than guess.
        if "neutral" in combined:
            best = "neutral"
        else:
            combined["neutral"] = 1.0 - max(combined.values())
            best = "neutral"
    return best, combined, kw_w, kw_evidence


def _validate_article_text(text: str) -> str | None:
    """Validate that the input looks like real news text, not gibberish."""
    words = re.findall(r"[a-zA-Z]{2,}", text)
    if len(words) < 5:
        return ("Please enter at least a full sentence (minimum 5 words). "
                "Single letters or short fragments cannot be analysed.")
    if sum(1 for c in text if c.isalpha()) / max(len(text), 1) < 0.40:
        return "The input appears to be mostly symbols or numbers. Please enter readable news text."

    # Use the trained TF-IDF vocabulary as a dictionary check:
    # gibberish words won't match any learned vocabulary features.
    if _models_loaded:
        X = _event_clf.vectorizer.transform([text])
        if X.nnz == 0:
            return ("The input doesn't contain recognisable financial or news terms. "
                    "Please paste a real news article or headline.")
    return None


def _top_k_event(combined: dict[str, float], k: int = 3) -> list[dict]:
    """Convert raw 7-class scores → frontend-id top-K entries (label + prob)."""
    from taxonomy.event_taxonomy import V2_TO_FRONTEND_EVENT as _EVT_MAP  # noqa: E402
    by_fe: dict[str, float] = {}
    for cls, prob in combined.items():
        fe_id = _EVT_MAP.get(cls, "market")
        by_fe[fe_id] = by_fe.get(fe_id, 0.0) + prob
    items = sorted(by_fe.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"id": i, "prob": round(p, 4)} for i, p in items]


def _top_k_stance(combined: dict[str, float]) -> list[dict]:
    items = sorted(
        ((V2_TO_FE_STANCE.get(s, "neutral"), p) for s, p in combined.items()),
        key=lambda kv: kv[1], reverse=True,
    )
    return [{"id": i, "prob": round(p, 4)} for i, p in items]


@app.post("/analyze")
def analyze_article(req: AnalyzeRequest):
    raw_text = (req.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if not _models_loaded:
        raise HTTPException(
            status_code=503,
            detail="ML models failed to load at startup. Analysis is unavailable.",
        )

    # Strip newsroom prefixes ("BT EXPLAINER |", "BREAKING:", "[Reuters]") and
    # other boilerplate that pollutes the TF-IDF vector for short inputs.
    cleaned = _clean_input_text(raw_text)

    # If either the raw or cleaned input matches a stored article, score the
    # FULL body so results match what the News Feed shows for the same article.
    # We try raw first because the stored headline may itself contain the
    # "BT EXPLAINER |" prefix that _clean_input_text strips out.
    matched = _lookup_stored_article(raw_text) or _lookup_stored_article(cleaned)
    if matched:
        text = f"{matched['headline']} {matched['clean_text']}".strip()
    else:
        text = cleaned

    err = _validate_article_text(text)
    if err:
        raise HTTPException(status_code=422, detail=err)

    from taxonomy.event_taxonomy import V2_TO_FRONTEND_EVENT as _EVT_MAP  # noqa: E402

    # --- Event classification (ensemble: ML + keyword rules) ---
    event_raw, event_combined, event_kw_w, event_evidence = _ensemble_event(text)
    event_conf = event_combined[event_raw]

    # 7-class problem: random baseline is ~0.14, calibrated reject at 0.18 lets
    # short but unambiguous headlines through (e.g. "Hertz files for chapter 11")
    # while still blocking truly noiseless input.
    if event_conf < 0.18:
        raise HTTPException(
            status_code=422,
            detail=("Could not confidently classify this text into any event category. "
                    "Please provide a more complete news article or headline."),
        )

    fe_event = _EVT_MAP.get(event_raw, "market")

    # --- Stance detection (ensemble: ML + keyword rules) ---
    stance_raw, stance_combined, stance_kw_w, stance_evidence = _ensemble_stance(text)
    stance_conf = stance_combined[stance_raw]
    fe_stance = V2_TO_FE_STANCE.get(stance_raw, "neutral")

    # --- Sub-scores (ML-predicted materiality, credibility, etc.) ---
    sub_scores = _sub_reg.predict_subscores(text)
    materiality = sub_scores["materiality"]
    credibility = sub_scores["credibility"]
    recency = 1.0  # live input = just submitted, maximum recency

    impact = calculate_impact_from_components(
        materiality=materiality,
        credibility=credibility,
        recency=recency,
    )

    matched_keywords = {
        "event":  event_evidence.get(event_raw, []),
        "stance": stance_evidence.get(stance_raw, []),
    }

    reasoning_bits = [
        f"event={event_raw} ({event_conf:.0%})",
        f"stance={stance_raw.upper()} ({stance_conf:.0%})",
    ]
    if matched_keywords["stance"]:
        reasoning_bits.append("matched: " + ", ".join(matched_keywords["stance"][:5]))
    reasoning = " · ".join(reasoning_bits) + f" · impact={impact:.3f}."

    return {
        "event_type":       fe_event,
        "stance":           fe_stance,
        "impact_score":     impact,
        "materiality":      materiality,
        "credibility":      credibility,
        "recency":          recency,
        "market_linkage":   sub_scores["market_linkage"],
        "time_sensitivity": sub_scores["time_sensitivity"],
        "event_confidence": round(event_conf, 4),
        "stance_confidence":round(stance_conf, 4),
        "event_top_k":      _top_k_event(event_combined),
        "stance_top_k":     _top_k_stance(stance_combined),
        "matched_keywords": matched_keywords,
        "matched_article":  bool(matched),
        "matched_article_id": matched["article_id"] if matched else None,
        "explanation":      reasoning,
    }


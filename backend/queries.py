"""
backend/queries.py
==================

Read-only query layer for the RNIA dashboard.

All queries:
- Open the SQLite DB in read-only immutable URI mode.
- Use parameterized queries (no f-string interpolation of user input).
- Return pandas DataFrames so the API layer can consume them directly.
- Pull from `articles JOIN v2_analyses` and ignore the v1 `old_*` columns.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH      = PROJECT_ROOT / "data" / "rnia.db"

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
EVENT_TYPES = [
    "Earnings", "Leadership_Change", "Regulatory_Action",
    "Mergers_Acquisitions", "Legal_Action", "Product_Announcement",
    "Macroeconomic_Geopolitical", "Market_Sentiment_Investor_Action",
    "Other", "Unclassified",
]
STANCES = ["BULLISH", "BEARISH", "NEUTRAL", "MIXED", "UNCLASSIFIED"]


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    """Open the DB in read-only mode."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB missing: {DB_PATH}")
    uri = f"file:{DB_PATH.resolve().as_posix()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=10)


# Common SELECT used everywhere — joined view of an analysed article.
# Note: COALESCE(published_at, scraped_at) is used downstream so that
# articles whose RSS feed didn't expose a pubDate still order correctly.
_BASE_SELECT = """
    SELECT
        a.article_id,
        a.headline,
        a.url,
        a.source,
        a.region,
        a.published_at,
        a.scraped_at,
        a.body_length,
        v.processing_status,
        v.filter_reason,
        v.relevance,
        v.event_type,
        v.stance,
        v.materiality,
        v.market_linkage,
        v.time_sensitivity,
        v.credibility,
        v.impact_score,
        v.reasoning,
        v.model_used,
        v.gemini_confidence_event,
        v.gemini_confidence_stance,
        v.escalation_reason
    FROM articles a
    JOIN v2_analyses v ON v.article_id = a.article_id
    WHERE v.processing_status = 'done'
      AND v.relevance = 1
"""


# ---------------------------------------------------------------------------
# 1) Latest articles (with optional event_type / min_impact filters)
# ---------------------------------------------------------------------------
def get_latest_articles(limit: int = 50,
                        event_type: str | None = None,
                        min_impact: float | None = None) -> pd.DataFrame:
    """Most recent analysed articles, newest first."""
    sql = _BASE_SELECT
    params: list = []
    if event_type:
        sql += " AND v.event_type = ?"
        params.append(event_type)
    if min_impact is not None:
        sql += " AND v.impact_score >= ?"
        params.append(float(min_impact))
    # Sort by best-known timestamp: real publish date if we have it,
    # otherwise the time we scraped it (so freshly-ingested articles
    # whose feed lacked a pubDate still appear at the top of the UI).
    sql += " ORDER BY COALESCE(a.published_at, a.scraped_at) DESC LIMIT ?"
    params.append(int(limit))
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# 2) Article detail (full row + sub-scores + raw escalator response)
# ---------------------------------------------------------------------------
def get_article_detail(article_id: int) -> pd.DataFrame:
    """Full row for one article. Returns a 1-row DataFrame (or empty)."""
    sql = """
        SELECT
            a.article_id,
            a.headline, a.url, a.source, a.region,
            a.published_at, a.scraped_at, a.body_length,
            a.clean_text, a.article_text,
            v.processing_status, v.filter_reason, v.error_message,
            v.relevance, v.event_type, v.stance,
            v.materiality, v.market_linkage,
            v.time_sensitivity, v.credibility, v.impact_score,
            v.reasoning, v.model_used,
            v.gemini_confidence_event, v.gemini_confidence_stance,
            v.escalation_reason,
            v.raw_gemini_response_json,
            v.raw_escalator_response_json,
            v.input_tokens, v.output_tokens, v.cost_usd,
            v.created_at, v.updated_at
        FROM articles a
        LEFT JOIN v2_analyses v ON v.article_id = a.article_id
        WHERE a.article_id = ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=[int(article_id)])


# ---------------------------------------------------------------------------
# 3) Filter by impact range
# ---------------------------------------------------------------------------
def filter_by_impact_range(low: float, high: float,
                           limit: int = 500) -> pd.DataFrame:
    sql = _BASE_SELECT + """
        AND v.impact_score BETWEEN ? AND ?
        ORDER BY v.impact_score DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn,
                                 params=[float(low), float(high), int(limit)])


# ---------------------------------------------------------------------------
# 4) Filter by event_type (list)
# ---------------------------------------------------------------------------
def filter_by_event_type(event_types: Iterable[str],
                         limit: int = 500) -> pd.DataFrame:
    types = [t for t in event_types if t in EVENT_TYPES]
    if not types:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(types))
    sql = _BASE_SELECT + f"""
        AND v.event_type IN ({placeholders})
        ORDER BY v.impact_score DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=[*types, int(limit)])


# ---------------------------------------------------------------------------
# 5) Filter by stance (list)
# ---------------------------------------------------------------------------
def filter_by_stance(stances: Iterable[str],
                     limit: int = 500) -> pd.DataFrame:
    s = [x for x in stances if x in STANCES]
    if not s:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(s))
    sql = _BASE_SELECT + f"""
        AND v.stance IN ({placeholders})
        ORDER BY v.impact_score DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=[*s, int(limit)])


# ---------------------------------------------------------------------------
# 6) Filter by date range  (ISO yyyy-mm-dd strings)
# ---------------------------------------------------------------------------
def filter_by_date_range(start: str, end: str,
                         limit: int = 1000) -> pd.DataFrame:
    sql = _BASE_SELECT + """
        AND a.published_at IS NOT NULL
        AND substr(a.published_at, 1, 10) BETWEEN ? AND ?
        ORDER BY a.published_at DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn,
                                 params=[str(start), str(end), int(limit)])


# ---------------------------------------------------------------------------
# 7) Summary stats — counts by event_type, stance, impact buckets
# ---------------------------------------------------------------------------
def summary_stats() -> dict[str, pd.DataFrame]:
    base_done = """
        FROM v2_analyses v
        WHERE v.processing_status = 'done' AND v.relevance = 1
    """
    sql_event = f"""
        SELECT v.event_type AS event_type, COUNT(*) AS n
        {base_done}
        GROUP BY v.event_type
        ORDER BY n DESC
    """
    sql_stance = f"""
        SELECT v.stance AS stance, COUNT(*) AS n
        {base_done}
        GROUP BY v.stance
        ORDER BY n DESC
    """
    sql_buckets = f"""
        SELECT
            CASE
                WHEN v.impact_score < 0.25 THEN '0.00-0.25'
                WHEN v.impact_score < 0.50 THEN '0.25-0.50'
                WHEN v.impact_score < 0.75 THEN '0.50-0.75'
                ELSE '0.75-1.00'
            END AS bucket,
            COUNT(*) AS n
        {base_done}
        GROUP BY bucket
        ORDER BY bucket
    """
    sql_overall = """
        SELECT
            (SELECT COUNT(*) FROM articles) AS total_articles,
            (SELECT COUNT(*) FROM v2_analyses
              WHERE processing_status='done')              AS analysed,
            (SELECT COUNT(*) FROM v2_analyses
              WHERE processing_status='done' AND relevance=1) AS relevant,
            (SELECT COUNT(*) FROM v2_analyses
              WHERE processing_status='filtered')          AS filtered,
            (SELECT COUNT(*) FROM v2_analyses
              WHERE processing_status='needs_escalation')  AS needs_escalation
    """
    with _connect() as conn:
        return {
            "event_type":      pd.read_sql_query(sql_event,    conn),
            "stance":          pd.read_sql_query(sql_stance,   conn),
            "impact_buckets":  pd.read_sql_query(sql_buckets,  conn),
            "overall":         pd.read_sql_query(sql_overall,  conn),
        }


# ---------------------------------------------------------------------------
# 8) Top articles by impact
# ---------------------------------------------------------------------------
def top_articles_by_impact(n: int = 20) -> pd.DataFrame:
    sql = _BASE_SELECT + """
        ORDER BY v.impact_score DESC, a.published_at DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=[int(n)])


# ---------------------------------------------------------------------------
# 9) Keyword search — LIKE on headline + clean_text
# ---------------------------------------------------------------------------
def search_by_keyword(text: str, limit: int = 200) -> pd.DataFrame:
    needle = (text or "").strip()
    if not needle:
        return pd.DataFrame()
    pat = f"%{needle.lower()}%"
    sql = _BASE_SELECT + """
        AND (LOWER(a.headline)   LIKE ?
             OR LOWER(a.clean_text) LIKE ?)
        ORDER BY v.impact_score DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=[pat, pat, int(limit)])


# ---------------------------------------------------------------------------
# 10) Escalated articles — Gemini Pro / Claude / "(corrected)" rows
# ---------------------------------------------------------------------------
def get_escalated_articles(limit: int = 500) -> pd.DataFrame:
    """
    Articles whose label was produced by the escalator path:
      * model_used contains 'pro'      (gemini-pro / gemini-3.1-pro-manual)
      * model_used contains 'claude'
      * model_used ends with '(corrected)'  (Stage-3 override)
    """
    sql = _BASE_SELECT + """
        AND (
            LOWER(v.model_used) LIKE '%pro%'
            OR LOWER(v.model_used) LIKE '%claude%'
            OR v.model_used LIKE '%(corrected)%'
        )
        ORDER BY a.published_at DESC
        LIMIT ?
    """
    with _connect() as conn:
        return pd.read_sql_query(sql, conn, params=[int(limit)])

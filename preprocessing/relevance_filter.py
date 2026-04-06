"""
relevance_filter.py — Step 1 (RNIA Spec): Relevance Filter
============================================================

Hard gate that drops articles with **no real economic / financial impact**
before they reach the event classifier.

Spec rejection rules:
    - Sports news (IPL, FIFA, cricket scores, ...)
    - Entertainment / gossip (Bollywood, box office, ...)
    - General news with no economic linkage

Spec acceptance rules:
    - Direct company impact (earnings, leadership, products, lawsuits)
    - Macro impact (inflation, war, trade, interest rates)
    - Policy / regulatory changes
    - Investor actions (hedge funds, ratings, large bets)

Approach (deterministic, model-free):
    1. Score the text on three axes — sports, entertainment, finance.
    2. Strong "single-hit" terms (e.g. "ipl", "bollywood") count for 3 points.
    3. Weaker generic terms count for 1 point each.
    4. Reuse the full ``EVENT_KEYWORDS`` vocabulary from
       ``annotation.auto_label_dataset`` as the finance signal — anything
       that would label as a finance event keeps the article.
    5. Reject only when sports/entertainment dominates AND the finance
       signal is weaker.
"""

from __future__ import annotations

import re
from typing import Tuple

from annotation.auto_label_dataset import EVENT_KEYWORDS


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Strong single-hit sports terms — one match is enough to count as sports.
STRONG_SPORTS = {
    "ipl", "fifa", "uefa", "bcci", "icc world cup", "world cup",
    "t20", "odi", "test match", "test cricket", "ranji",
    "premier league", "la liga", "bundesliga", "serie a",
    "champions league", "europa league", "epl",
    "nba", "nhl", "mlb", "nfl",
    "wimbledon", "australian open", "french open", "us open tennis",
    "grand slam", "grand prix", "formula 1", "formula one", "f1 race",
    "olympics", "olympic games", "commonwealth games", "asian games",
    "wicket", "wickets", "batsman", "batter", "bowler", "all-rounder",
    "hat-trick", "century scored", "half-century",
    "dream11", "fantasy cricket",
}

# Weaker sports terms — need 2+ hits (or 1 strong + 1 weak) to count.
WEAK_SPORTS = {
    "cricket", "football", "soccer", "rugby", "baseball", "hockey match",
    "tournament", "league match", "playoff", "playoffs", "semi-final",
    "quarter-final", "final match", "knockout",
    "coach", "captain", "squad", "lineup", "starting xi",
    "goal scored", "goalkeeper", "striker", "midfielder", "defender",
    "umpire", "referee", "pitch", "stadium",
    "innings", "over by over", "boundary", "sixer", "no-ball",
    "dugout", "training camp",
    "kohli", "rohit sharma", "dhoni", "messi", "ronaldo", "neymar",
    "shubman gill", "vaibhav sooryavanshi", "sachin tendulkar",
    "vs", "highlights",
}

# Strong single-hit entertainment terms.
STRONG_ENTERTAINMENT = {
    "bollywood", "hollywood", "tollywood", "kollywood",
    "box office collection", "box office", "first day collection",
    "weekend collection", "lifetime collection",
    "oscar", "oscars", "academy award", "grammy", "grammys",
    "filmfare", "iifa", "cannes film festival",
    "movie review", "film review",
}

# Weaker entertainment terms.
WEAK_ENTERTAINMENT = {
    "movie", "film", "actor", "actress", "celebrity", "celeb",
    "concert", "album", "song", "music video", "single release",
    "cinema", "theatre release", "trailer",
    "web series", "netflix original", "amazon prime original",
    "director", "co-star", "leading lady",
    "dance", "singer",
}

# Macro / geopolitical terms that the spec explicitly treats as financial
# (e.g., "war / negotiations = Macroeconomic / Geopolitical").
MACRO_POLITICAL = {
    "war", "ceasefire", "peace talks", "peace deal", "diplomacy",
    "sanctions", "embargo", "trade deal", "trade war", "tariff war",
    "geopolitical", "sovereign debt", "supply shock",
    "energy crisis", "oil supply", "opec", "opec+",
    "election results", "policy reform", "fiscal stimulus",
    # Commodities / FX (daily-price coverage is still financial)
    "gold rate", "gold rates", "silver rate", "silver rates",
    "bullion", "bullion market", "commodity prices",
    "rupee rate", "dollar rate", "usd-inr", "usd inr",
    "petrol price", "diesel price", "fuel price",
}

# Common business / finance vocabulary — single-token signals that an
# article is about business or markets. Kept conservative (no political
# generic terms like "minister" alone) to avoid false positives on
# pure political content.
COMMON_BUSINESS = {
    "investment", "investments", "investor", "investors",
    "company", "companies", "firm", "firms", "corporate",
    "industry", "industries", "sector", "sectors",
    "business", "businesses",
    "economy", "economic", "financial", "finance",
    "shares", "share", "stock", "stocks",
    "market", "markets", "trading", "trader", "traders",
    "exchange", "exchanges", "valuation", "valuations",
    "crore", "crores", "billion", "million", "lakh", "lakhs",
    "deal", "deals", "fund", "funds", "funding", "capital",
    "subsidy", "scheme", "schemes", "incentive", "incentives",
    "growth", "demand", "supply",
    "production", "manufacturing", "manufacturer", "manufacturers",
    "factory", "plant", "capacity", "output",
    "import", "imports", "export", "exports",
    "consumer", "consumers", "consumption",
    "auto", "automaker", "automotive", "automobile",
    "telecom", "pharma", "pharmaceutical", "infrastructure",
    "real estate", "realty", "housing market",
    "airline", "airlines", "shipping",
    "energy", "renewable", "solar", "wind power", "battery",
    "ev", "electric vehicle", "semiconductor", "chip",
    "ai", "artificial intelligence", "cloud", "cybersecurity",
}

# Finance signal — reuse the full event-keyword vocabulary so that
# anything labelable as a finance event keeps the article.
FINANCE_KEYWORDS: set[str] = set()
for _kws in EVENT_KEYWORDS.values():
    FINANCE_KEYWORDS.update(k.lower() for k in _kws)
FINANCE_KEYWORDS.update(MACRO_POLITICAL)
FINANCE_KEYWORDS.update(COMMON_BUSINESS)

# A small set of "single-hit strong finance" terms that, even alone,
# unambiguously signal a financial article.
STRONG_FINANCE = {
    "earnings", "revenue", "ebitda", "ipo", "merger", "acquisition",
    "stock price", "share price", "nasdaq", "nyse", "sensex", "nifty",
    "dow jones", "s&p 500", "quarterly results", "eps", "dividend",
    "fiscal", "sec filing", "fda approval", "fed rate", "rate hike",
    "rate cut", "interest rate", "inflation", "gdp", "tariff",
    "central bank", "regulator", "antitrust", "lawsuit",
    "ceo", "cfo",
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_WORD_BOUNDARY = re.compile(r"[a-z0-9][a-z0-9\-]*")

# Cache compiled patterns per vocabulary id() to avoid recompiling on every
# call. Each term is matched with `\b` word boundaries so that short tokens
# like "ipl" don't accidentally match inside "diplomatic".
_PATTERN_CACHE: dict[int, re.Pattern] = {}


def _compile_vocab(vocabulary: set[str]) -> re.Pattern:
    """Return a compiled regex that matches any term in *vocabulary* as a whole word/phrase."""
    cache_key = id(vocabulary)
    cached = _PATTERN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Sort by length desc so longer multi-word phrases match before substrings.
    parts = [re.escape(t) for t in sorted(vocabulary, key=len, reverse=True)]
    pattern = re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)
    _PATTERN_CACHE[cache_key] = pattern
    return pattern


def _count_hits(text: str, vocabulary: set[str]) -> int:
    """Return the number of distinct vocabulary terms present in *text* (word-boundary matched)."""
    if not text or not vocabulary:
        return 0
    pattern = _compile_vocab(vocabulary)
    found = {m.group(0).lower() for m in pattern.finditer(text)}
    return len(found)


def _has_any(text: str, vocabulary: set[str]) -> bool:
    """Return True if any vocabulary term appears in *text* as a whole word/phrase."""
    if not text or not vocabulary:
        return False
    return _compile_vocab(vocabulary).search(text) is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def relevance_score(headline: str, body: str) -> dict:
    """
    Compute the (sports, entertainment, finance) scores for an article.

    Strong terms count as 3 points, weak terms as 1 point each.

    Parameters
    ----------
    headline : str
        Article headline (may be empty).
    body : str
        Cleaned article body text (may be empty).

    Returns
    -------
    dict
        ``{"sports": int, "entertainment": int, "finance": int}``
    """
    text = f"{headline or ''} {body or ''}".lower()

    sports = (3 if _has_any(text, STRONG_SPORTS) else 0) + _count_hits(text, WEAK_SPORTS)
    ent    = (3 if _has_any(text, STRONG_ENTERTAINMENT) else 0) + _count_hits(text, WEAK_ENTERTAINMENT)
    fin    = (3 if _has_any(text, STRONG_FINANCE) else 0) + _count_hits(text, FINANCE_KEYWORDS)

    return {"sports": sports, "entertainment": ent, "finance": fin}


def is_relevant(headline: str, body: str) -> Tuple[bool, str]:
    """
    Decide whether an article has real financial / economic impact.

    Decision logic (headline gets the final say when it dominates):
        1. **Headline veto** — if the headline alone has a strong sports
           or entertainment hit AND the headline finance score is weak
           (< 3), reject. A movie/sports story is still that story even
           if its body is full of business words.
        2. Combined-text fallback — sports / ent score >= 3 AND
           dominates finance → reject.
        3. No finance signal at all → reject.
        4. Text too short → reject.
        5. Else → RELEVANT.

    Parameters
    ----------
    headline : str
        Article headline.
    body : str
        Cleaned article body text.

    Returns
    -------
    tuple[bool, str]
        ``(is_relevant, reason)`` — reason is a short human-readable string.
    """
    text = f"{headline or ''} {body or ''}".strip()
    word_count = len(_WORD_BOUNDARY.findall(text.lower()))
    if word_count < 8:
        return False, "insufficient text"

    # Headline-only veto — strong sports/ent in the title with no real
    # finance hook in the title kills the article regardless of body.
    h_scores = relevance_score(headline, "")
    h_sp, h_en, h_fn = h_scores["sports"], h_scores["entertainment"], h_scores["finance"]
    if h_sp >= 3 and h_fn < 3:
        return False, f"sports headline (h_sports={h_sp}, h_finance={h_fn})"
    if h_en >= 3 and h_fn < 3:
        return False, f"entertainment headline (h_ent={h_en}, h_finance={h_fn})"

    scores = relevance_score(headline, body)
    sp, en, fn = scores["sports"], scores["entertainment"], scores["finance"]

    if sp >= 3 and sp > fn:
        return False, f"sports content (sports={sp}, finance={fn})"
    if en >= 3 and en > fn:
        return False, f"entertainment content (entertainment={en}, finance={fn})"
    if fn == 0 and (sp > 0 or en > 0):
        return False, "non-financial content with no finance signal"
    if fn == 0:
        return False, "no financial signal detected"

    return True, "relevant"


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        ("Rajasthan Royals vs Royal Challengers Bengaluru Live Score, IPL 2026: RCB 6 wickets down", ""),
        ("Apple reports record quarterly earnings, beats estimates", "revenue growth strong"),
        ("Has India ever played in a FIFA World Cup?", "the answer might surprise you"),
        ("Dhurandhar 2 box office collection day 32: Aditya Dhar's blockbuster", "earned crores"),
        ("Fed cuts interest rates by 25 basis points amid easing inflation", "central bank decision"),
        ("Reliance Industries posts 12% rise in Q3 net profit", "revenue beats consensus"),
        ("Virat Kohli writes a note to RR's star batter Vaibhav Sooryavanshi", ""),
        ("Israel-Hamas peace talks resume in Cairo", "diplomatic effort to end the war"),
    ]
    for h, b in samples:
        ok, reason = is_relevant(h, b)
        flag = "OK  " if ok else "DROP"
        print(f"[{flag}] {h[:80]:80s}  →  {reason}")

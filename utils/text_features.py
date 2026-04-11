"""
text_features.py — Shared input-text construction for RNIA models
==================================================================

Single source of truth for converting (headline, body) → model input text.
Used by training, evaluation, and inference paths so feature distributions
stay consistent.

Headline weighting
------------------
Headlines are usually more discriminative per token than article bodies
(they pre-summarise the financial event). Weighting them N times during
training boosts their TF-IDF signal:

    text = (headline + " ") * N + clean_text

The weight is persisted in ``models/saved_models/text_features_config.json``
when training, and read back at inference time so models trained with
weight=N see the same text construction at predict time.

Default weight is 1 (no weighting) so existing pre-trained models keep
working unchanged. Bump to 3 only when retraining.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
TEXT_FEATURES_CONFIG_PATH = os.path.join(SAVED_MODELS_DIR, "text_features_config.json")

DEFAULT_HEADLINE_WEIGHT = 1   # legacy / no weighting
RECOMMENDED_HEADLINE_WEIGHT = 3


def _load_config() -> dict:
    if os.path.isfile(TEXT_FEATURES_CONFIG_PATH):
        try:
            with open(TEXT_FEATURES_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read text_features_config.json: %s", e)
    return {}


def get_headline_weight(default: int = DEFAULT_HEADLINE_WEIGHT) -> int:
    """Return the headline weight stored at training time, else *default*."""
    cfg = _load_config()
    weight = int(cfg.get("headline_weight", default))
    return max(1, weight)


def save_text_features_config(headline_weight: int) -> None:
    """Persist text-feature settings so inference paths can read them."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    cfg = {"headline_weight": int(headline_weight)}
    with open(TEXT_FEATURES_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Saved text_features_config: %s", cfg)


# ---------------------------------------------------------------------------
# Text construction
# ---------------------------------------------------------------------------

def build_input_text(
    headline: Optional[str],
    body: Optional[str],
    weight: Optional[int] = None,
) -> str:
    """
    Build the canonical model input string from a (headline, body) pair.

    The headline is repeated *weight* times before being concatenated with
    the body — so if weight=3, each headline word effectively counts 3×
    in the TF-IDF feature vector.

    Parameters
    ----------
    headline : str | None
        Article headline. Empty/None is treated as no headline.
    body : str | None
        Article body or cleaned text.
    weight : int | None
        Headline repetition count. If None, reads from saved config
        (defaults to 1).

    Returns
    -------
    str
        Combined text ready for vectorizer.fit_transform / transform.
    """
    if weight is None:
        weight = get_headline_weight()
    weight = max(1, int(weight))

    h = (headline or "").strip()
    b = (body or "").strip()

    if not h and not b:
        return ""
    if not h:
        return b
    if not b:
        return h
    if weight <= 1:
        return f"{h} {b}"
    # Repeat headline `weight` times so its tokens dominate the TF vector
    return (h + " ") * weight + b


def build_input_texts(
    headlines: list[str] | list[Optional[str]],
    bodies: list[str] | list[Optional[str]],
    weight: Optional[int] = None,
) -> list[str]:
    """Vectorised version of build_input_text for training pipelines."""
    if weight is None:
        weight = get_headline_weight()
    return [build_input_text(h, b, weight=weight) for h, b in zip(headlines, bodies)]


def build_input_text_from_row(
    row,
    weight: Optional[int] = None,
) -> str:
    """
    Convenience helper for pandas rows (e.g. df.apply(build_input_text_from_row, axis=1)).

    Looks for `headline` and `clean_text` columns; falls back to `article_text`
    when `clean_text` is not present.
    """
    headline = row.get("headline") if hasattr(row, "get") else getattr(row, "headline", None)
    body = (
        row.get("clean_text") if hasattr(row, "get") else getattr(row, "clean_text", None)
    )
    if not body:
        body = (
            row.get("article_text") if hasattr(row, "get")
            else getattr(row, "article_text", None)
        )
    return build_input_text(headline, body, weight=weight)

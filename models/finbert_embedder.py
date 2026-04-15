"""
finbert_embedder.py — FinBERT [CLS] embeddings for feature augmentation
========================================================================

Extracts 768-dim [CLS] pooled embeddings from FinBERT (ProsusAI/finbert)
for each article. Used by ``event_classifier_hybrid`` to concatenate
semantic embeddings with TF-IDF features → LR classifier.

Why this design
---------------
- Keeps interpretability: the classifier head is still LR (coefficients
  per feature dimension are inspectable). FinBERT only contributes
  features, not predictions.
- Cacheable: extracting embeddings for ~2k articles takes ~2-5 min on
  RTX 3050. We cache to disk so retraining doesn't repay the cost.
- Optional: only loaded when needed (lazy import of torch/transformers).

Cache file
----------
``models/saved_models/finbert_embedding_cache.joblib`` — dict keyed by
SHA-256 hash of the input text.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
EMBEDDING_CACHE_PATH = os.path.join(SAVED_MODELS_DIR, "finbert_embedding_cache.joblib")

FINBERT_MODEL = "ProsusAI/finbert"
EMBEDDING_DIM = 768
MAX_TOKEN_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


def _hash_text(text: str) -> str:
    """Stable SHA-256 hash for cache keying."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class FinBERTEmbedder:
    """
    Lazy-loaded FinBERT [CLS] embedding extractor with persistent cache.

    Usage
    -----
    >>> emb = FinBERTEmbedder()
    >>> X_emb = emb.extract_batch(texts)   # (n, 768) float32 numpy array
    >>> emb.save_cache()
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE,
                 cache_path: str = EMBEDDING_CACHE_PATH):
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._tokenizer = None
        self._model = None
        self._device = None
        self._cache: dict[str, np.ndarray] = {}
        self._cache_dirty = False
        self._load_cache()

    # ------------------------------------------------------------------
    # Lazy initialisation of the heavy parts
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers + torch required for FinBERT embeddings. Install:\n"
                "    pip install transformers torch"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("FinBERT embedder using GPU: %s (%.1f GB)",
                        torch.cuda.get_device_name(0), mem_gb)
        else:
            logger.info("FinBERT embedder using CPU (slow but works).")

        logger.info("Loading FinBERT (%s) for embedding extraction...", FINBERT_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self._model = AutoModel.from_pretrained(FINBERT_MODEL).to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    def _load_cache(self) -> None:
        if os.path.isfile(self.cache_path):
            try:
                self._cache = joblib.load(self.cache_path)
                logger.info("FinBERT embedding cache loaded: %d entries (%s)",
                            len(self._cache), self.cache_path)
            except (OSError, ValueError, EOFError) as e:
                logger.warning("Failed to load embedding cache: %s. Starting fresh.", e)
                self._cache = {}
        else:
            self._cache = {}

    def save_cache(self) -> None:
        if not self._cache_dirty:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        joblib.dump(self._cache, self.cache_path)
        logger.info("FinBERT embedding cache saved: %d entries", len(self._cache))
        self._cache_dirty = False

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Run FinBERT on a list of texts and return (n, 768) [CLS] embeddings."""
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            encoded = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=MAX_TOKEN_LENGTH, return_tensors="pt",
            ).to(self._device)
            outputs = self._model(**encoded)
            # Use [CLS] token embedding (first token of last hidden state)
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        return cls

    def extract_embedding(self, text: str) -> np.ndarray:
        """Single-text embedding (uses cache when possible)."""
        key = _hash_text(text)
        if key in self._cache:
            return self._cache[key]
        emb = self._embed_batch([text])[0]
        self._cache[key] = emb
        self._cache_dirty = True
        return emb

    def extract_batch(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Batched extraction with caching. Returns (n, 768) ndarray aligned with input order.
        """
        n = len(texts)
        out = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

        # First pass: split into cached / pending
        pending_idx: list[int] = []
        pending_texts: list[str] = []
        for i, t in enumerate(texts):
            key = _hash_text(t)
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                pending_idx.append(i)
                pending_texts.append(t)

        if not pending_idx:
            logger.info("FinBERT embeddings: all %d texts hit the cache.", n)
            return out

        logger.info("FinBERT embeddings: %d/%d texts need extraction (cache hit %d).",
                    len(pending_idx), n, n - len(pending_idx))

        # Batched inference for the misses
        for batch_start in range(0, len(pending_idx), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pending_idx))
            batch_texts = pending_texts[batch_start:batch_end]
            batch_indices = pending_idx[batch_start:batch_end]

            embeds = self._embed_batch(batch_texts)
            for offset, orig_idx in enumerate(batch_indices):
                out[orig_idx] = embeds[offset]
                self._cache[_hash_text(pending_texts[batch_start + offset])] = embeds[offset]

            if show_progress:
                done = batch_end
                logger.info("  FinBERT extraction: %d/%d", done, len(pending_idx))

        self._cache_dirty = True
        return out

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @property
    def device(self) -> Optional[str]:
        return self._device

    def __len__(self) -> int:
        return len(self._cache)

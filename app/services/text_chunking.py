"""Deterministic, bounded text chunking for the embedding model (pure Python, no model needed).

all-MiniLM-L6-v2 silently truncates its input at ~256 word-pieces (~180 English words), so long sections
(e.g. a resume's experience or the JD) would otherwise lose everything after the first paragraph.

Policy
  * words are whitespace tokens; windows of `max_words` advance by `max_words - overlap_words`
  * at most `max_chunks` windows are kept (the START of the text is kept; `truncated=True` is reported)
  * empty / whitespace-only text -> no chunks
  * chunk weights = word counts, used for length-weighted mean pooling (see aggregate_vectors)
"""
from typing import List, Sequence, Tuple

import numpy as np


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 30, max_chunks: int = 8) -> Tuple[List[str], bool]:
    """Returns (chunks, truncated)."""
    if max_words < 1 or not (0 <= overlap_words < max_words) or max_chunks < 1:
        raise ValueError("need max_words >= 1, 0 <= overlap_words < max_words, max_chunks >= 1")
    words = (text or "").split()
    if not words:
        return [], False
    if len(words) <= max_words:
        return [" ".join(words)], False
    step = max_words - overlap_words
    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_words]))
        if start + max_words >= len(words):
            break
        start += step
    truncated = len(chunks) > max_chunks
    return chunks[:max_chunks], truncated


def aggregate_vectors(vectors: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Length-weighted mean of chunk vectors, re-normalised to unit L2 norm (zero vector if nothing to aggregate).
    Mean pooling keeps the section representation a single vector, so storage and cosine scoring are unchanged."""
    if len(vectors) == 0:
        raise ValueError("no vectors to aggregate")
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D stack of vectors, got shape {arr.shape}")
    w = np.ones(len(arr), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    if w.shape != (len(arr),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative, one per vector, with a positive sum")
    mean = (arr * w[:, None]).sum(axis=0) / w.sum()
    n = float(np.linalg.norm(mean))
    return (mean / n).astype(np.float32) if n > 0 else mean.astype(np.float32)

"""
IMGNet Metrics
==============

Generic similarity metrics that can be applied to any 1-D embedding vectors.

Notes:
- All functions expect 1-D numpy arrays or array-like sequences of the same length.
- Window size and thresholds follow the IMG paper defaults:
    window_size = 11
    threshold   = 8
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Defaults exposed from the paper / experiments
WINDOW_SIZE: int = 11
THRESHOLD: int = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_1d(a: Sequence[float]) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 2:
        raise ValueError("Embedding must contain at least 2 elements.")
    return arr


def _signs(a: np.ndarray) -> np.ndarray:
    return np.where(a >= 0, 1, -1).astype(np.int8)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def img_sign_score(
    e1: Sequence[float],
    e2: Sequence[float],
    *,
    window_size: int = WINDOW_SIZE,
    threshold: int = THRESHOLD,
    beta: float = 10.0,
) -> float:
    """
    IMG Sign Score

    Measures local relational sign agreement between two embeddings by sliding
    a fixed-length window and checking whether at least `threshold` dimensions
    share the same sign.

    Returns a score in [0, 1].
    """
    e1 = _as_1d(e1)
    e2 = _as_1d(e2)

    if e1.shape != e2.shape:
        raise ValueError("Embeddings must have the same length.")

    n = e1.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError("Embedding too short for the given window size.")

    # Soft agreement per dimension, then thresholded per-window
    a = (np.tanh(beta * e1 * e2) + 1.0) / 2.0

    # cumulative sum trick for sliding window sum
    c = np.cumsum(np.insert(a, 0, 0.0))
    s = c[window_size:] - c[:-window_size]

    m = s >= (threshold - 0.5)  # differentiable-ish gate approximation
    return float(np.mean(m))


def amp_img_score(
    e1: Sequence[float],
    e2: Sequence[float],
    *,
    window_size: int = WINDOW_SIZE,
    threshold: int = THRESHOLD,
) -> float:
    """
    AMP IMG Score

    Extends IMG Sign with amplitude consistency: only windows that pass the
    sign threshold are additionally checked for local mean-magnitude similarity.

    Returns a score in [0, 1].
    """
    e1 = _as_1d(e1)
    e2 = _as_1d(e2)

    if e1.shape != e2.shape:
        raise ValueError("Embeddings must have the same length.")

    n = e1.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError("Embedding too short for the given window size.")

    s1 = _signs(e1)
    s2 = _signs(e2)

    c1 = np.cumsum(np.insert(np.abs(e1), 0, 0.0))
    c2 = np.cumsum(np.insert(np.abs(e2), 0, 0.0))
    a1 = (c1[window_size:] - c1[:-window_size]) / window_size
    a2 = (c2[window_size:] - c2[:-window_size]) / window_size

    match = (s1[:-window_size + 1] == s2[:-window_size + 1]).astype(np.float64)
    for i in range(1, window_size):
        match += (s1[i : i + n] == s2[i : i + n]).astype(np.float64)

    passed = match >= threshold
    if not np.any(passed):
        return 0.0

    base = np.maximum(a1[passed], a2[passed])
    amp_sim = np.clip(1.0 - np.abs(a1[passed] - a2[passed]) / np.maximum(base, 1e-6), 0.0, 1.0)
    return float(np.mean(amp_sim))


def chain_score(
    e1: Sequence[float],
    e2: Sequence[float],
    *,
    window_size: int = WINDOW_SIZE,
    threshold: int = THRESHOLD,
    neutral_len: int = 29,
    reward_rate: float = 0.3,
    punish_rate: float = 1.0,
) -> tuple[float, int, float]:
    """
    Chain Score

    Encourages contiguous chains of matching windows. Longer chains indicate
    stronger identity consistency.

    Returns:
        score  : float in [0, 1]
        chains : int, number of matching chains
        avg_chain : float, average chain length
    """
    e1 = _as_1d(e1)
    e2 = _as_1d(e2)

    if e1.shape != e2.shape:
        raise ValueError("Embeddings must have the same length.")

    n = e1.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError("Embedding too short for the given window size.")

    flags = []
    for i in range(n):
        s1 = _signs(e1[i : i + window_size])
        s2 = _signs(e2[i : i + window_size])
        flags.append(int(np.sum(s1 == s2)) >= threshold)

    flags = np.asarray(flags, dtype=bool)
    total = int(np.sum(flags))
    img_s = total / max(n, 1)

    chains = 0
    in_c = False
    for f in flags:
        if f and not in_c:
            chains += 1
            in_c = True
        elif not f:
            in_c = False

    if chains == 0 or total == 0:
        return 0.0, 0, 0.0

    avg_c = total / chains
    diff = avg_c - neutral_len
    score = img_s + ((reward_rate * diff) if diff >= 0 else (punish_rate * diff)) / 100.0
    return float(np.clip(score, 0.0, 1.0)), chains, float(avg_c)


def cosine_similarity(
    e1: Sequence[float],
    e2: Sequence[float],
) -> float:
    """
    Cosine similarity baseline for comparison with IMG metrics.

    Returns a score in [-1, 1].
    """
    e1 = _as_1d(e1)
    e2 = _as_1d(e2)

    if e1.shape != e2.shape:
        raise ValueError("Embeddings must have the same length.")

    return float(
        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
    )


# ---------------------------------------------------------------------------
# Voting / batch helpers
# ---------------------------------------------------------------------------

VOTE_MATCH = "MATCH"
VOTE_UNCERTAIN = "UNCERTAIN"
VOTE_DIFFERENT = "DIFFERENT"


def _vote(img_sign_val: float, amp_val: float, chain_val: float, threshold: float = 0.79) -> str:
    n_pass = sum(v >= threshold for v in (img_sign_val, amp_val, chain_val))
    if n_pass >= 2:
        return VOTE_MATCH
    if n_pass == 1:
        return VOTE_UNCERTAIN
    return VOTE_DIFFERENT


def batch_compare(
    e1: Sequence[float],
    e2: Sequence[float],
    *,
    threshold: float = 0.79,
    window_size: int = WINDOW_SIZE,
    threshold_window: int = THRESHOLD,
) -> dict:
    """
    Compare two embeddings with all available metrics and produce a verdict.

    Returns a dict with per-metric scores and a final vote.
    """
    img = img_sign_score(e1, e2, window_size=window_size, threshold=threshold_window)
    amp = amp_img_score(e1, e2, window_size=window_size, threshold=threshold_window)
    ch, chains, avg_chain = chain_score(
        e1, e2, window_size=window_size, threshold=threshold_window
    )
    cos = cosine_similarity(e1, e2)

    return {
        "img_sign": img,
        "amp_img": amp,
        "chain_score": ch,
        "cosine": cos,
        "vote": _vote(img, amp, ch, threshold=threshold),
        "chains": chains,
        "avg_chain": avg_chain,
    }

"""
Liveness detection prototype for ImgNet demo.

This is a lightweight heuristic-based liveness check, not a full anti-spoofing
model. It is intended as a demonstrator for how IMG metrics could be used in
a liveness workflow.

Heuristic:
- Require two images from the same identity.
- Compute IMG metrics + cosine.
- A live pair tends to show:
  - MATCH vote
  - moderate-to-high cosine
  - strong chain score
  - stable img_sign under small synthetic perturbations
- A spoof/presentation attack tends to show:
  - MATCH or UNCERTAIN vote with suspiciously high cosine but low chain/IMG sign
  - larger drop in img_sign under perturbation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from imgnet.metrics import batch_compare, img_sign_score
from imgnet.visualizer import compute_embedding


@dataclass
class LivenessResult:
    live: bool
    confidence: float
    vote: str
    img_sign: float
    cosine: float
    chain_score: float
    chains: int
    img_sign_delta: float
    reason: str


def _perturb(emb: np.ndarray, rng: np.random.Generator, severity: float = 0.1) -> np.ndarray:
    out = emb.copy()
    n = max(1, int(severity * out.shape[0]))
    idx = rng.choice(out.shape[0], size=n, replace=False)
    out[idx] += rng.normal(0, severity * out.std(), size=n)
    return out


def check_liveness(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    seed: int = 42,
    perturb_severity: float = 0.1,
) -> LivenessResult:
    rng = np.random.default_rng(seed)
    emb_a = compute_embedding(img_a)
    emb_b = compute_embedding(img_b)
    result = batch_compare(emb_a, emb_b)

    img_sign_orig = result["img_sign"]
    img_sign_p = img_sign_score(_perturb(emb_a, rng, perturb_severity), _perturb(emb_b, rng, perturb_severity))
    delta = img_sign_orig - img_sign_p

    vote = result["vote"]
    cos = result["cosine"]
    chain = result["chain_score"]
    chains = result["chains"]

    confidence = 0.0
    reasons = []

    if vote == "MATCH":
        confidence += 0.4
        reasons.append("metrics agree")
    elif vote == "UNCERTAIN":
        confidence += 0.1
        reasons.append("uncertain metrics")
    else:
        reasons.append("metrics disagree")

    if cos > 0.7:
        confidence += 0.2
        reasons.append("high cosine")
    elif cos > 0.4:
        confidence += 0.1
        reasons.append("moderate cosine")

    if chain > 0.7:
        confidence += 0.25
        reasons.append("strong chain")
    elif chain > 0.4:
        confidence += 0.1
        reasons.append("moderate chain")

    if chains >= 5:
        confidence += 0.15
        reasons.append("multiple chains")

    if delta < 0.1:
        confidence += 0.15
        reasons.append("stable under perturbation")
    elif delta > 0.3:
        confidence -= 0.2
        reasons.append("unstable under perturbation")

    confidence = float(np.clip(confidence, 0.0, 1.0))
    live = confidence >= 0.6
    return LivenessResult(
        live=live,
        confidence=confidence,
        vote=vote,
        img_sign=img_sign_orig,
        cosine=cos,
        chain_score=chain,
        chains=chains,
        img_sign_delta=delta,
        reason="; ".join(reasons) if reasons else "insufficient signal",
    )

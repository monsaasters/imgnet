"""
IMGNet Occlusion Benchmark
==========================

Synthetic occlusion benchmark to compare IMG metrics vs cosine similarity.

Workflow:
1. Load embeddings from an existing benchmark dataset or a local CSV/NPY.
2. For each pair, generate occlusion variants by zeroing selected dimensions.
3. Recompute embeddings or directly perturb the embedding vectors.
4. Compare metric stability under perturbation.
5. Export CSV/JSON summary.

This is intended to demonstrate that IMG Sign is more robust to
embedding-space perturbations than cosine similarity, without retraining
the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from imgnet.metrics import (
    amp_img_score,
    batch_compare,
    chain_score,
    cosine_similarity,
    img_sign_score,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingPair:
    pair_id: str
    label: str  # "same" or "different"
    embedding_a: np.ndarray
    embedding_b: np.ndarray


@dataclass
class PerturbationResult:
    pair_id: str
    label: str
    strategy: str
    severity: float
    img_sign: float
    amp_img: float
    chain_score: float
    cosine: float
    vote: str
    img_sign_delta: float
    cosine_delta: float


# ---------------------------------------------------------------------------
# Synthetic perturbations on embedding space
# ---------------------------------------------------------------------------

def _perturb_random(
    emb: np.ndarray,
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    """Add Gaussian noise to a subset of dimensions."""
    out = emb.copy()
    n = max(1, int(severity * out.shape[0]))
    idx = rng.choice(out.shape[0], size=n, replace=False)
    out[idx] += rng.normal(0, severity * out.std(), size=n)
    return out


def _perturb_block(
    emb: np.ndarray,
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    """Zero out a contiguous block of dimensions."""
    out = emb.copy()
    block = max(1, int(severity * out.shape[0]))
    start = int(rng.integers(0, max(1, out.shape[0] - block)))
    out[start : start + block] = 0.0
    return out


def _perturb_amplitude(
    emb: np.ndarray,
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    """Scale the magnitude of selected dimensions."""
    out = emb.copy()
    n = max(1, int(severity * out.shape[0]))
    idx = rng.choice(out.shape[0], size=n, replace=False)
    scale = rng.uniform(0.0, max(0.01, 1.0 - severity), size=n)
    out[idx] *= scale
    return out


PERTURBATIONS = {
    "random_noise": _perturb_random,
    "block_zero": _perturb_block,
    "amplitude_drop": _perturb_amplitude,
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    window_size: int = 11
    threshold_window: int = 8
    vote_threshold: float = 0.79
    severities: Sequence[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    strategies: Sequence[str] = field(default_factory=lambda: list(PERTURBATIONS.keys()))
    seed: int = 42


def _score_pair(
    a: np.ndarray,
    b: np.ndarray,
    cfg: BenchmarkConfig,
) -> tuple[float, float, float, float, str]:
    img = img_sign_score(a, b, window_size=cfg.window_size, threshold=cfg.threshold_window)
    amp = amp_img_score(a, b, window_size=cfg.window_size, threshold=cfg.threshold_window)
    ch, _, _ = chain_score(a, b, window_size=cfg.window_size, threshold=cfg.threshold_window)
    cos = cosine_similarity(a, b)
    vote = batch_compare(a, b, threshold=cfg.vote_threshold, window_size=cfg.window_size, threshold_window=cfg.threshold_window)["vote"]
    return img, amp, ch, cos, vote


def run_benchmark(
    pairs: Iterable[EmbeddingPair],
    cfg: BenchmarkConfig | None = None,
) -> list[PerturbationResult]:
    if cfg is None:
        cfg = BenchmarkConfig()

    rng = np.random.default_rng(cfg.seed)
    results: list[PerturbationResult] = []

    for pair in pairs:
        img_orig, amp_orig, ch_orig, cos_orig, _ = _score_pair(pair.embedding_a, pair.embedding_b, cfg)

        for strategy in cfg.strategies:
            fn = PERTURBATIONS[strategy]
            for severity in cfg.severities:
                a_p = fn(pair.embedding_a, rng, severity)
                b_p = fn(pair.embedding_b, rng, severity)

                img, amp, ch, cos, vote = _score_pair(a_p, b_p, cfg)

                results.append(
                    PerturbationResult(
                        pair_id=pair.pair_id,
                        label=pair.label,
                        strategy=strategy,
                        severity=severity,
                        img_sign=img,
                        amp_img=amp,
                        chain_score=ch,
                        cosine=cos,
                        vote=vote,
                        img_sign_delta=img_orig - img,
                        cosine_delta=cos_orig - cos,
                    )
                )

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_pairs_from_npy_dir(
    path: Path,
    pattern: str = "*.npy",
) -> list[EmbeddingPair]:
    """
    Load pairs from a directory where filenames encode pair identity.

    Expected filename format:
        <pair_id>_A.npy
        <pair_id>_B.npy

    Optional sidecar: <pair_id>.json with {"label": "same|different"}
    """
    pairs: list[EmbeddingPair] = []
    files = sorted(path.glob(pattern))
    base_names: dict[str, Path] = {}
    for f in files:
        stem = f.stem
        if stem.endswith("_A") or stem.endswith("_B"):
            base = stem[:-2]
            base_names.setdefault(base, {})[stem[-1]] = f

    for base, parts in base_names.items():
        if "A" not in parts or "B" not in parts:
            continue
        a = np.load(parts["A"]).reshape(-1)
        b = np.load(parts["B"]).reshape(-1)
        label = "same"
        meta = path / f"{base}.json"
        if meta.exists():
            try:
                label = json.loads(meta.read_text(encoding="utf-8")).get("label", label)
            except Exception:
                pass
        pairs.append(EmbeddingPair(pair_id=base, label=label, embedding_a=a, embedding_b=b))

    return pairs


def load_pairs_from_csv(
    path: Path,
    embedding_cols: Sequence[int],
    label_col: int | None = None,
    delimiter: str = ",",
) -> list[EmbeddingPair]:
    """
    Load pairs from a CSV file.

    Expected format: each row is one embedding vector. Pairing is implicit:
    rows 0 & 1 -> pair 0, rows 2 & 3 -> pair 1, etc.

    If label_col is given, it is read from the first row of each pair and
    ignored for the second row.
    """
    data = np.genfromtxt(path, delimiter=delimiter, comments=None)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    pairs: list[EmbeddingPair] = []
    it = iter(range(0, data.shape[0] - 1, 2))
    for i in it:
        if i + 1 >= data.shape[0]:
            break
        a = data[i, list(embedding_cols)].astype(np.float64)
        b = data[i + 1, list(embedding_cols)].astype(np.float64)
        label = "same"
        if label_col is not None:
            label = str(data[i, label_col])
        pairs.append(EmbeddingPair(pair_id=f"row_{i}", label=label, embedding_a=a, embedding_b=b))

    return pairs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(results: list[PerturbationResult]) -> dict:
    if not results:
        return {}

    arr = np.array(
        [
            [r.severity, r.img_sign, r.cosine, r.img_sign_delta, r.cosine_delta]
            for r in results
        ]
    )

    summary = {
        "total_results": len(results),
        "avg_img_sign": float(np.mean([r.img_sign for r in results])),
        "avg_cosine": float(np.mean([r.cosine for r in results])),
        "avg_img_sign_delta": float(np.mean([r.img_sign_delta for r in results])),
        "avg_cosine_delta": float(np.mean([r.cosine_delta for r in results])),
        "vote_counts": {k: sum(1 for r in results if r.vote == k) for k in ["MATCH", "UNCERTAIN", "DIFFERENT"]},
        "by_strategy": {},
    }

    for strategy in sorted({r.strategy for r in results}):
        subset = [r for r in results if r.strategy == strategy]
        summary["by_strategy"][strategy] = {
            "avg_img_sign": float(np.mean([r.img_sign for r in subset])),
            "avg_cosine": float(np.mean([r.cosine for r in subset])),
            "avg_img_sign_delta": float(np.mean([r.img_sign_delta for r in subset])),
            "avg_cosine_delta": float(np.mean([r.cosine_delta for r in subset])),
        }

    return summary


def to_csv(results: list[PerturbationResult], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "label",
                "strategy",
                "severity",
                "img_sign",
                "amp_img",
                "chain_score",
                "cosine",
                "vote",
                "img_sign_delta",
                "cosine_delta",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "pair_id": r.pair_id,
                    "label": r.label,
                    "strategy": r.strategy,
                    "severity": r.severity,
                    "img_sign": f"{r.img_sign:.6f}",
                    "amp_img": f"{r.amp_img:.6f}",
                    "chain_score": f"{r.chain_score:.6f}",
                    "cosine": f"{r.cosine:.6f}",
                    "vote": r.vote,
                    "img_sign_delta": f"{r.img_sign_delta:.6f}",
                    "cosine_delta": f"{r.cosine_delta:.6f}",
                }
            )


def to_json(results: list[PerturbationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize(results),
        "results": [
            {
                "pair_id": r.pair_id,
                "label": r.label,
                "strategy": r.strategy,
                "severity": r.severity,
                "img_sign": r.img_sign,
                "amp_img": r.amp_img,
                "chain_score": r.chain_score,
                "cosine": r.cosine,
                "vote": r.vote,
                "img_sign_delta": r.img_sign_delta,
                "cosine_delta": r.cosine_delta,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

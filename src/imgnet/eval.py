"""
IMGNet evaluation utilities

Evaluate an IMGNet checkpoint against reference embeddings or a dataset.
Produces comparison tables and can compare:
- Pure IMG Sign model
- Hybrid IMG+Cosine model
- Cosine-only baseline
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from imgnet.metrics import batch_compare, cosine_similarity, img_sign_score


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    sample_id: str
    label: str  # "same" or "different"
    embedding_a: np.ndarray
    embedding_b: np.ndarray


@dataclass
class EvalResult:
    sample_id: str
    label: str
    img_sign: float
    amp_img: float
    chain_score: float
    cosine: float
    vote: str
    correct: bool


@dataclass
class EvalConfig:
    threshold: float = 0.79
    window_size: int = 11
    threshold_window: int = 8
    seed: int = 42


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_samples(samples: Sequence[Sample], cfg: EvalConfig | None = None) -> list[EvalResult]:
    if cfg is None:
        cfg = EvalConfig()

    results: list[EvalResult] = []
    for sample in samples:
        result = batch_compare(
            sample.embedding_a,
            sample.embedding_b,
            threshold=cfg.threshold,
            window_size=cfg.window_size,
            threshold_window=cfg.threshold_window,
        )
        vote = result["vote"]
        correct = (vote == "MATCH" and sample.label == "same") or (vote != "MATCH" and sample.label == "different")
        results.append(
            EvalResult(
                sample_id=sample.sample_id,
                label=sample.label,
                img_sign=result["img_sign"],
                amp_img=result["amp_img"],
                chain_score=result["chain_score"],
                cosine=result["cosine"],
                vote=vote,
                correct=bool(correct),
            )
        )
    return results


def summarize_eval(results: Sequence[EvalResult]) -> dict:
    if not results:
        return {}
    accuracy = sum(1 for r in results if r.correct) / len(results)
    same_results = [r for r in results if r.label == "same"]
    diff_results = [r for r in results if r.label == "different"]
    return {
        "total": len(results),
        "accuracy": accuracy,
        "same_count": len(same_results),
        "different_count": len(diff_results),
        "vote_counts": {k: sum(1 for r in results if r.vote == k) for k in ["MATCH", "UNCERTAIN", "DIFFERENT"]},
        "avg_img_sign": float(np.mean([r.img_sign for r in results])),
        "avg_cosine": float(np.mean([r.cosine for r in results])),
        "accuracy_same": sum(1 for r in same_results if r.correct) / max(len(same_results), 1),
        "accuracy_different": sum(1 for r in diff_results if r.correct) / max(len(diff_results), 1),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_samples_from_npy_dir(path: Path, pattern: str = "*.npy") -> list[Sample]:
    files = sorted(path.glob(pattern))
    base_map: dict[str, dict[str, Path]] = {}
    for f in files:
        stem = f.stem
        if stem.endswith("_A") or stem.endswith("_B"):
            base = stem[:-2]
            base_map.setdefault(base, {})[stem[-1]] = f

    samples: list[Sample] = []
    for base, parts in base_map.items():
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
        samples.append(Sample(sample_id=base, label=label, embedding_a=a, embedding_b=b))
    return samples


def load_samples_from_csv(path: Path, embedding_cols: Sequence[int], label_col: Optional[int] = None, delimiter: str = ",") -> list[Sample]:
    data = np.genfromtxt(path, delimiter=delimiter, comments=None)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    samples: list[Sample] = []
    it = iter(range(0, data.shape[0] - 1, 2))
    for i in it:
        if i + 1 >= data.shape[0]:
            break
        a = data[i, list(embedding_cols)].astype(np.float64)
        b = data[i + 1, list(embedding_cols)].astype(np.float64)
        label = "same"
        if label_col is not None:
            label = str(data[i, label_col])
        samples.append(Sample(sample_id=f"row_{i}", label=label, embedding_a=a, embedding_b=b))
    return samples


def results_to_csv(results: Sequence[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "label", "img_sign", "amp_img", "chain_score", "cosine", "vote", "correct"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "label": r.label,
                    "img_sign": f"{r.img_sign:.6f}",
                    "amp_img": f"{r.amp_img:.6f}",
                    "chain_score": f"{r.chain_score:.6f}",
                    "cosine": f"{r.cosine:.6f}",
                    "vote": r.vote,
                    "correct": int(r.correct),
                }
            )


def results_to_json(results: Sequence[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_eval(results),
        "results": [
            {
                "sample_id": r.sample_id,
                "label": r.label,
                "img_sign": r.img_sign,
                "amp_img": r.amp_img,
                "chain_score": r.chain_score,
                "cosine": r.cosine,
                "vote": r.vote,
                "correct": r.correct,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

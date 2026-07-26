"""
IMGNet Occlusion Robustness Benchmark by Mode
==============================================

Compare how IMG-only, cosine-only, and hybrid modes degrade under
synthetic embedding-space occlusion.

Workflow:
1. Load embedding pairs.
2. For each severity level, perturb embeddings with multiple strategies.
3. Evaluate each mode's accuracy after perturbation.
4. Export CSV/JSON showing robustness per mode.

This benchmark is designed to demonstrate that IMG metrics can be more
stable than cosine under embedding perturbations, without retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from imgnet.eval import EvalConfig, Sample, load_samples_from_csv, load_samples_from_npy_dir
from imgnet.metrics import cosine_similarity, img_sign_score


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

def _perturb_random(emb: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    out = emb.copy()
    n = max(1, int(severity * out.shape[0]))
    idx = rng.choice(out.shape[0], size=n, replace=False)
    out[idx] += rng.normal(0, severity * out.std(), size=n)
    return out


def _perturb_block(emb: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    out = emb.copy()
    block = max(1, int(severity * out.shape[0]))
    start = int(rng.integers(0, max(1, out.shape[0] - block)))
    out[start : start + block] = 0.0
    return out


def _perturb_amplitude(emb: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
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
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RobustnessResult:
    sample_id: str
    label: str
    strategy: str
    severity: float
    mode: str
    img_sign: float
    cosine: float
    hybrid_score: float
    vote: str
    correct: bool


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------

def hybrid_score(img_sign: float, cosine: float, weight: float = 0.5) -> float:
    return weight * img_sign + (1.0 - weight) * (cosine + 1.0) / 2.0


def hybrid_vote(img_sign: float, cosine: float, threshold: float = 0.79, weight: float = 0.5) -> str:
    score = hybrid_score(img_sign, cosine, weight=weight)
    if score >= threshold:
        return "MATCH"
    if score >= threshold - 0.15:
        return "UNCERTAIN"
    return "DIFFERENT"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RobustnessConfig:
    threshold: float = 0.79
    hybrid_weight: float = 0.5
    window_size: int = 11
    threshold_window: int = 8
    severities: Sequence[float] = (0.05, 0.1, 0.2, 0.4)
    strategies: Sequence[str] = ("random_noise", "block_zero", "amplitude_drop")
    seed: int = 42


def _vote_mode(img_sign: float, cosine: float, mode: str, cfg: RobustnessConfig) -> str:
    if mode == "img_only":
        return "MATCH" if img_sign >= cfg.threshold else ("UNCERTAIN" if img_sign >= cfg.threshold - 0.15 else "DIFFERENT")
    if mode == "cosine_only":
        cos_norm = (cosine + 1.0) / 2.0
        return "MATCH" if cos_norm >= cfg.threshold else ("UNCERTAIN" if cos_norm >= cfg.threshold - 0.15 else "DIFFERENT")
    return hybrid_vote(img_sign, cosine, threshold=cfg.threshold, weight=cfg.hybrid_weight)


def run_robustness(
    samples: Sequence[Sample],
    cfg: RobustnessConfig | None = None,
) -> list[RobustnessResult]:
    if cfg is None:
        cfg = RobustnessConfig()

    rng = np.random.default_rng(cfg.seed)
    results: list[RobustnessResult] = []

    for sample in samples:
        for strategy in cfg.strategies:
            fn = PERTURBATIONS[strategy]
            for severity in cfg.severities:
                a_p = fn(sample.embedding_a, rng, severity)
                b_p = fn(sample.embedding_b, rng, severity)

                img = img_sign_score(a_p, b_p, window_size=cfg.window_size, threshold=cfg.threshold_window)
                cos = cosine_similarity(a_p, b_p)
                h_score = hybrid_score(img, cos, weight=cfg.hybrid_weight)

                for mode in ("img_only", "cosine_only", "hybrid"):
                    vote = _vote_mode(img, cos, mode, cfg)
                    correct = (vote == "MATCH" and sample.label == "same") or (vote != "MATCH" and sample.label == "different")
                    results.append(
                        RobustnessResult(
                            sample_id=sample.sample_id,
                            label=sample.label,
                            strategy=strategy,
                            severity=severity,
                            mode=mode,
                            img_sign=img,
                            cosine=cos,
                            hybrid_score=h_score,
                            vote=vote,
                            correct=bool(correct),
                        )
                    )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_robustness(results: Sequence[RobustnessResult]) -> dict:
    summary: dict = {}
    for mode in ("img_only", "cosine_only", "hybrid"):
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
        summary[mode] = {
            "total": len(mode_results),
            "overall_accuracy": sum(1 for r in mode_results if r.correct) / len(mode_results),
            "by_strategy": {},
        }
        for strategy in sorted({r.strategy for r in mode_results}):
            subset = [r for r in mode_results if r.strategy == strategy]
            summary[mode]["by_strategy"][strategy] = {
                "total": len(subset),
                "accuracy_by_severity": {},
            }
            for severity in sorted({r.severity for r in subset}):
                sev_subset = [r for r in subset if r.severity == severity]
                summary[mode]["by_strategy"][strategy]["accuracy_by_severity"][str(severity)] = (
                    sum(1 for r in sev_subset if r.correct) / len(sev_subset)
                )
    return summary


def results_to_csv_robustness(results: Sequence[RobustnessResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id", "label", "strategy", "severity", "mode",
                "img_sign", "cosine", "hybrid_score", "vote", "correct",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "label": r.label,
                    "strategy": r.strategy,
                    "severity": r.severity,
                    "mode": r.mode,
                    "img_sign": f"{r.img_sign:.6f}",
                    "cosine": f"{r.cosine:.6f}",
                    "hybrid_score": f"{r.hybrid_score:.6f}",
                    "vote": r.vote,
                    "correct": int(r.correct),
                }
            )


def results_to_json_robustness(results: Sequence[RobustnessResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_robustness(results),
        "results": [
            {
                "sample_id": r.sample_id,
                "label": r.label,
                "strategy": r.strategy,
                "severity": r.severity,
                "mode": r.mode,
                "img_sign": r.img_sign,
                "cosine": r.cosine,
                "hybrid_score": r.hybrid_score,
                "vote": r.vote,
                "correct": r.correct,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="IMGNet occlusion robustness benchmark by mode")
    ap.add_argument("--input", type=Path, required=True, help="Input directory (.npy) or CSV file")
    ap.add_argument("--format", choices=["npy_dir", "csv"], default="npy_dir")
    ap.add_argument("--csv-col", type=int, nargs="+", default=None, help="Embedding columns for CSV")
    ap.add_argument("--label-col", type=int, default=None, help="Label column for CSV")
    ap.add_argument("--threshold", type=float, default=0.79)
    ap.add_argument("--hybrid-weight", type=float, default=0.5)
    ap.add_argument("--window-size", type=int, default=11)
    ap.add_argument("--threshold-window", type=int, default=8)
    ap.add_argument("--output-csv", type=Path, default=Path("robustness_by_mode.csv"))
    ap.add_argument("--output-json", type=Path, default=Path("robustness_by_mode.json"))
    args = ap.parse_args(argv)

    if args.format == "npy_dir":
        samples = load_samples_from_npy_dir(args.input)
    else:
        if args.csv_col is None:
            print("ERROR: --csv-col is required for CSV format", file=sys.stderr)
            return 1
        samples = load_samples_from_csv(args.input, embedding_cols=args.csv_col, label_col=args.label_col)

    if not samples:
        print("ERROR: No samples loaded.", file=sys.stderr)
        return 1

    cfg = RobustnessConfig(
        threshold=args.threshold,
        hybrid_weight=args.hybrid_weight,
        window_size=args.window_size,
        threshold_window=args.threshold_window,
    )
    results = run_robustness(samples, cfg)
    summary = summarize_robustness(results)

    results_to_csv_robustness(results, args.output_csv)
    results_to_json_robustness(results, args.output_json)

    print(f"Loaded {len(samples)} pairs")
    print(f"Total results: {len(results)}")
    print("\nSummary:")
    for mode, stats in summary.items():
        print(f"  {mode}: accuracy={stats['overall_accuracy']:.2%} ({stats['total']} samples)")
    print(f"\nSaved: {args.output_csv}, {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

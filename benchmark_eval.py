"""
IMGNet Evaluation Benchmark
===========================

Compare similarity modes on a set of embedding pairs:
- IMG Sign only
- Cosine only
- Hybrid: weighted combination of IMG Sign + Cosine

Outputs CSV/JSON with per-sample and summary results.
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

from imgnet.eval import EvalConfig, Sample, evaluate_samples, load_samples_from_csv, load_samples_from_npy_dir, results_to_csv, results_to_json, summarize_eval
from imgnet.metrics import batch_compare, cosine_similarity, img_sign_score


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------

def hybrid_score(img_sign: float, cosine: float, weight: float = 0.5) -> float:
    """Combine IMG Sign and Cosine into a single similarity score."""
    return weight * img_sign + (1.0 - weight) * (cosine + 1.0) / 2.0


def hybrid_vote(img_sign: float, cosine: float, threshold: float = 0.79, weight: float = 0.5) -> str:
    """Vote based on hybrid score."""
    score = hybrid_score(img_sign, cosine, weight=weight)
    if score >= threshold:
        return "MATCH"
    if score >= threshold - 0.15:
        return "UNCERTAIN"
    return "DIFFERENT"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    sample_id: str
    label: str
    mode: str
    img_sign: float
    cosine: float
    hybrid_score: float
    vote: str
    correct: bool


def run_benchmark(
    samples: Sequence[Sample],
    *,
    threshold: float = 0.79,
    hybrid_weight: float = 0.5,
    cfg: EvalConfig | None = None,
) -> list[BenchmarkResult]:
    if cfg is None:
        cfg = EvalConfig(threshold=threshold)

    results: list[BenchmarkResult] = []
    for sample in samples:
        img = img_sign_score(sample.embedding_a, sample.embedding_b, window_size=cfg.window_size, threshold=cfg.threshold_window)
        cos = cosine_similarity(sample.embedding_a, sample.embedding_b)

        # Mode 1: IMG only
        img_only_vote = "MATCH" if img >= threshold else ("UNCERTAIN" if img >= threshold - 0.15 else "DIFFERENT")
        img_only_correct = (img_only_vote == "MATCH" and sample.label == "same") or (img_only_vote != "MATCH" and sample.label == "different")

        results.append(BenchmarkResult(
            sample_id=sample.sample_id,
            label=sample.label,
            mode="img_only",
            img_sign=img,
            cosine=cos,
            hybrid_score=0.0,
            vote=img_only_vote,
            correct=img_only_correct,
        ))

        # Mode 2: Cosine only
        cos_norm = (cos + 1.0) / 2.0
        cos_only_vote = "MATCH" if cos_norm >= threshold else ("UNCERTAIN" if cos_norm >= threshold - 0.15 else "DIFFERENT")
        cos_only_correct = (cos_only_vote == "MATCH" and sample.label == "same") or (cos_only_vote != "MATCH" and sample.label == "different")

        results.append(BenchmarkResult(
            sample_id=sample.sample_id,
            label=sample.label,
            mode="cosine_only",
            img_sign=img,
            cosine=cos,
            hybrid_score=0.0,
            vote=cos_only_vote,
            correct=cos_only_correct,
        ))

        # Mode 3: Hybrid
        h_score = hybrid_score(img, cos, weight=hybrid_weight)
        h_vote = hybrid_vote(img, cos, threshold=threshold, weight=hybrid_weight)
        h_correct = (h_vote == "MATCH" and sample.label == "same") or (h_vote != "MATCH" and sample.label == "different")

        results.append(BenchmarkResult(
            sample_id=sample.sample_id,
            label=sample.label,
            mode="hybrid",
            img_sign=img,
            cosine=cos,
            hybrid_score=h_score,
            vote=h_vote,
            correct=h_correct,
        ))

    return results


def summarize_benchmark(results: Sequence[BenchmarkResult]) -> dict:
    summary = {}
    for mode in ["img_only", "cosine_only", "hybrid"]:
        mode_results = [r for r in results if r.mode == mode]
        if not mode_results:
            continue
        summary[mode] = {
            "total": len(mode_results),
            "accuracy": sum(1 for r in mode_results if r.correct) / len(mode_results),
            "vote_counts": {k: sum(1 for r in mode_results if r.vote == k) for k in ["MATCH", "UNCERTAIN", "DIFFERENT"]},
            "avg_img_sign": float(np.mean([r.img_sign for r in mode_results])),
            "avg_cosine": float(np.mean([r.cosine for r in mode_results])),
            "avg_hybrid_score": float(np.mean([r.hybrid_score for r in mode_results])),
        }
    return summary


def results_to_csv_benchmark(results: Sequence[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "label", "mode", "img_sign", "cosine", "hybrid_score", "vote", "correct"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "label": r.label,
                    "mode": r.mode,
                    "img_sign": f"{r.img_sign:.6f}",
                    "cosine": f"{r.cosine:.6f}",
                    "hybrid_score": f"{r.hybrid_score:.6f}",
                    "vote": r.vote,
                    "correct": int(r.correct),
                }
            )


def results_to_json_benchmark(results: Sequence[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_benchmark(results),
        "results": [
            {
                "sample_id": r.sample_id,
                "label": r.label,
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
    ap = argparse.ArgumentParser(description="IMGNet evaluation benchmark")
    ap.add_argument("--input", type=Path, required=True, help="Input directory (.npy) or CSV file")
    ap.add_argument("--format", choices=["npy_dir", "csv"], default="npy_dir")
    ap.add_argument("--csv-col", type=int, nargs="+", default=None, help="Embedding columns for CSV")
    ap.add_argument("--label-col", type=int, default=None, help="Label column for CSV")
    ap.add_argument("--threshold", type=float, default=0.79)
    ap.add_argument("--hybrid-weight", type=float, default=0.5)
    ap.add_argument("--output-csv", type=Path, default=Path("benchmark_eval.csv"))
    ap.add_argument("--output-json", type=Path, default=Path("benchmark_eval.json"))
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

    results = run_benchmark(samples, threshold=args.threshold, hybrid_weight=args.hybrid_weight)
    summary = summarize_benchmark(results)

    results_to_csv_benchmark(results, args.output_csv)
    results_to_json_benchmark(results, args.output_json)

    print(f"Loaded {len(samples)} pairs")
    print(f"Total results: {len(results)}")
    print("\nSummary:")
    for mode, stats in summary.items():
        print(f"  {mode}: accuracy={stats['accuracy']:.2%} ({stats['total']} samples)")
    print(f"\nSaved: {args.output_csv}, {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

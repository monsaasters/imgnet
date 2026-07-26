"""
IMGNet Threshold Sweep
======================

Find the optimal threshold for each similarity mode by evaluating a range
of thresholds on a validation set.

Modes:
- img_only
- cosine_only
- hybrid

Outputs:
- CSV with threshold, accuracy, vote counts per mode
- JSON summary with best thresholds
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from imgnet.eval import EvalConfig, Sample, evaluate_samples, load_samples_from_csv, load_samples_from_npy_dir, summarize_eval
from imgnet.metrics import cosine_similarity, img_sign_score


def sweep_thresholds(
    samples: Sequence[Sample],
    *,
    thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.79, 0.8, 0.85, 0.9, 0.95),
    hybrid_weight: float = 0.5,
    window_size: int = 11,
    threshold_window: int = 8,
) -> list[dict]:
    results = []
    for threshold in thresholds:
        img_correct = 0
        cos_correct = 0
        hybrid_correct = 0
        total = 0
        img_votes = {"MATCH": 0, "UNCERTAIN": 0, "DIFFERENT": 0}
        cos_votes = {"MATCH": 0, "UNCERTAIN": 0, "DIFFERENT": 0}
        hybrid_votes = {"MATCH": 0, "UNCERTAIN": 0, "DIFFERENT": 0}

        for sample in samples:
            img = img_sign_score(sample.embedding_a, sample.embedding_b, window_size=window_size, threshold=threshold_window)
            cos = cosine_similarity(sample.embedding_a, sample.embedding_b)
            cos_norm = (cos + 1.0) / 2.0
            hybrid_score = hybrid_weight * img + (1.0 - hybrid_weight) * cos_norm

            # IMG only vote
            img_vote = "MATCH" if img >= threshold else ("UNCERTAIN" if img >= threshold - 0.15 else "DIFFERENT")
            img_correct += ((img_vote == "MATCH" and sample.label == "same") or (img_vote != "MATCH" and sample.label == "different"))
            img_votes[img_vote] += 1

            # Cosine only vote
            cos_vote = "MATCH" if cos_norm >= threshold else ("UNCERTAIN" if cos_norm >= threshold - 0.15 else "DIFFERENT")
            cos_correct += ((cos_vote == "MATCH" and sample.label == "same") or (cos_vote != "MATCH" and sample.label == "different"))
            cos_votes[cos_vote] += 1

            # Hybrid vote
            hybrid_vote = "MATCH" if hybrid_score >= threshold else ("UNCERTAIN" if hybrid_score >= threshold - 0.15 else "DIFFERENT")
            hybrid_correct += ((hybrid_vote == "MATCH" and sample.label == "same") or (hybrid_vote != "MATCH" and sample.label == "different"))
            hybrid_votes[hybrid_vote] += 1

            total += 1

        results.append({
            "threshold": threshold,
            "img_only_accuracy": img_correct / max(total, 1),
            "cosine_only_accuracy": cos_correct / max(total, 1),
            "hybrid_accuracy": hybrid_correct / max(total, 1),
            "img_only_votes": img_votes,
            "cosine_only_votes": cos_votes,
            "hybrid_votes": hybrid_votes,
            "total": total,
        })

    return results


def hybrid_score(img_sign: float, cosine: float, weight: float = 0.5) -> float:
    return weight * img_sign + (1.0 - weight) * (cosine + 1.0) / 2.0


def summarize_sweep(results: Sequence[dict]) -> dict:
    best_img = max(results, key=lambda x: x["img_only_accuracy"])
    best_cos = max(results, key=lambda x: x["cosine_only_accuracy"])
    best_hybrid = max(results, key=lambda x: x["hybrid_accuracy"])
    return {
        "best_img_threshold": best_img["threshold"],
        "best_img_accuracy": best_img["img_only_accuracy"],
        "best_cos_threshold": best_cos["threshold"],
        "best_cos_accuracy": best_cos["cosine_only_accuracy"],
        "best_hybrid_threshold": best_hybrid["threshold"],
        "best_hybrid_accuracy": best_hybrid["hybrid_accuracy"],
        "results": list(results),
    }


def save_sweep_csv(results: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "img_only_accuracy",
                "cosine_only_accuracy",
                "hybrid_accuracy",
                "img_only_MATCH",
                "img_only_UNCERTAIN",
                "img_only_DIFFERENT",
                "cosine_only_MATCH",
                "cosine_only_UNCERTAIN",
                "cosine_only_DIFFERENT",
                "hybrid_MATCH",
                "hybrid_UNCERTAIN",
                "hybrid_DIFFERENT",
                "total",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "threshold": r["threshold"],
                "img_only_accuracy": f"{r['img_only_accuracy']:.4f}",
                "cosine_only_accuracy": f"{r['cosine_only_accuracy']:.4f}",
                "hybrid_accuracy": f"{r['hybrid_accuracy']:.4f}",
                "img_only_MATCH": r["img_only_votes"]["MATCH"],
                "img_only_UNCERTAIN": r["img_only_votes"]["UNCERTAIN"],
                "img_only_DIFFERENT": r["img_only_votes"]["DIFFERENT"],
                "cosine_only_MATCH": r["cosine_only_votes"]["MATCH"],
                "cosine_only_UNCERTAIN": r["cosine_only_votes"]["UNCERTAIN"],
                "cosine_only_DIFFERENT": r["cosine_only_votes"]["DIFFERENT"],
                "hybrid_MATCH": r["hybrid_votes"]["MATCH"],
                "hybrid_UNCERTAIN": r["hybrid_votes"]["UNCERTAIN"],
                "hybrid_DIFFERENT": r["hybrid_votes"]["DIFFERENT"],
                "total": r["total"],
            })


def save_sweep_json(results: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize_sweep(results)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="IMGNet threshold sweep")
    ap.add_argument("--input", type=Path, required=True, help="Input directory (.npy) or CSV file")
    ap.add_argument("--format", choices=["npy_dir", "csv"], default="npy_dir")
    ap.add_argument("--csv-col", type=int, nargs="+", default=None, help="Embedding columns for CSV")
    ap.add_argument("--label-col", type=int, default=None, help="Label column for CSV")
    ap.add_argument("--hybrid-weight", type=float, default=0.5)
    ap.add_argument("--window-size", type=int, default=11)
    ap.add_argument("--threshold-window", type=int, default=8)
    ap.add_argument("--output-csv", type=Path, default=Path("threshold_sweep.csv"))
    ap.add_argument("--output-json", type=Path, default=Path("threshold_sweep.json"))
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

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.79, 0.8, 0.85, 0.9, 0.95]
    results = sweep_thresholds(
        samples,
        thresholds=thresholds,
        hybrid_weight=args.hybrid_weight,
        window_size=args.window_size,
        threshold_window=args.threshold_window,
    )
    summary = summarize_sweep(results)

    save_sweep_csv(results, args.output_csv)
    save_sweep_json(results, args.output_json)

    print(f"Loaded {len(samples)} pairs")
    print("\nBest thresholds:")
    print(f"  IMG only  : {summary['best_img_threshold']:.2f} -> {summary['best_img_accuracy']:.2%}")
    print(f"  Cosine    : {summary['best_cos_threshold']:.2f} -> {summary['best_cos_accuracy']:.2%}")
    print(f"  Hybrid    : {summary['best_hybrid_threshold']:.2f} -> {summary['best_hybrid_accuracy']:.2%}")
    print(f"\nSaved: {args.output_csv}, {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

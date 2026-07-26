"""
CLI for IMGNet similarity metrics.

Examples:
    imgnet verify a.jpg b.jpg
    imgnet verify a.jpg b.jpg --metric img_sign
    imgnet verify a.jpg b.jpg --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from imgnet.metrics import (
    batch_compare,
    cosine_similarity,
    VOTE_MATCH,
    VOTE_UNCERTAIN,
    VOTE_DIFFERENT,
)


def _load_embedding(path: Path) -> list[float]:
    """
    Placeholder loader.

    Accepts:
    - .npy file produced by numpy.save
    - .json file containing a list/array
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".npy":
        import numpy as np
        arr = np.load(path)
        return arr.reshape(-1).tolist()

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "embedding" in data:
                data = data["embedding"]
            elif "vector" in data:
                data = data["vector"]
            else:
                raise ValueError("JSON must contain 'embedding' or 'vector'.")
        if not isinstance(data, list):
            raise ValueError("Embedding JSON must be a list.")
        return [float(x) for x in data]

    raise ValueError(f"Unsupported embedding format: {path.suffix}")


def _verdict_label(vote: str) -> str:
    if vote == VOTE_MATCH:
        return "MATCH"
    if vote == VOTE_UNCERTAIN:
        return "UNCERTAIN"
    if vote == VOTE_DIFFERENT:
        return "DIFFERENT"
    return vote


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="imgnet")
    sub = ap.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="Verify similarity between two embeddings")
    verify.add_argument("a", type=Path, help="Embedding file A (.npy/.json)")
    verify.add_argument("b", type=Path, help="Embedding file B (.npy/.json)")
    verify.add_argument(
        "--metric",
        default="auto",
        choices=["auto", "img_sign", "amp_img", "chain_score", "cosine"],
        help="Metric to use; auto = full voting",
    )
    verify.add_argument("--json", action="store_true", help="Print JSON output")

    args = ap.parse_args(argv)

    e1 = _load_embedding(args.a)
    e2 = _load_embedding(args.b)

    if args.metric == "auto":
        result = batch_compare(e1, e2)
        vote = result["vote"]
        label = _verdict_label(vote)

        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"IMG Sign : {result['img_sign']:.4f}")
            print(f"AMP IMG  : {result['amp_img']:.4f}")
            print(f"Chain    : {result['chain_score']:.4f}")
            print(f"Cosine   : {result['cosine']:.4f}")
            print(f"Chains   : {result['chains']}")
            print(f"AvgChain : {result['avg_chain']:.4f}")
            print(f"Verdict  : {label}")
        return 0

    if args.metric == "img_sign":
        from imgnet.metrics import img_sign_score
        score = img_sign_score(e1, e2)
    elif args.metric == "amp_img":
        from imgnet.metrics import amp_img_score
        score = amp_img_score(e1, e2)
    elif args.metric == "chain_score":
        from imgnet.metrics import chain_score as chain_fn
        score, chains, avg_chain = chain_fn(e1, e2)
    elif args.metric == "cosine":
        score = cosine_similarity(e1, e2)
    else:
        raise ValueError(args.metric)

    chains_out = 0
    avg_chain_out = 0.0
    if args.metric == "chain_score":
        score, chains_out, avg_chain_out = chain_fn(e1, e2)

    if args.json:
        out = {"metric": args.metric, "score": float(score)}
        if args.metric == "chain_score":
            out["chains"] = int(chains_out)
            out["avg_chain"] = float(avg_chain_out)
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"{args.metric}: {float(score):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

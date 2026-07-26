"""
Example: using IMGNet metrics on generic embeddings.

This does NOT require the IMGNet face model. It shows that the metrics work
on any 1-D embedding vectors: text, audio, recommendation, synthetic, etc.
"""

from __future__ import annotations

import numpy as np

from imgnet.metrics import (
    batch_compare,
    cosine_similarity,
    img_sign_score,
)


def main() -> None:
    # Example 1: synthetic embeddings
    rng = np.random.default_rng(42)
    e1 = rng.standard_normal(1024)
    e2 = e1 + rng.normal(0, 0.1, 1024)

    print("=== Example 1: near-duplicate embeddings ===")
    print(f"cosine    : {cosine_similarity(e1, e2):.4f}")
    print(f"img_sign  : {img_sign_score(e1, e2):.4f}")
    print(f"batch     : {batch_compare(e1, e2)}")

    # Example 2: opposite embeddings
    e3 = -e1
    print("\n=== Example 2: opposite embeddings ===")
    print(f"cosine    : {cosine_similarity(e1, e3):.4f}")
    print(f"img_sign  : {img_sign_score(e1, e3):.4f}")
    print(f"batch     : {batch_compare(e1, e3)}")

    # Example 3: random embeddings
    e4 = rng.standard_normal(1024)
    e5 = rng.standard_normal(1024)
    print("\n=== Example 3: random embeddings ===")
    print(f"cosine    : {cosine_similarity(e4, e5):.4f}")
    print(f"img_sign  : {img_sign_score(e4, e5):.4f}")
    print(f"batch     : {batch_compare(e4, e5)}")


if __name__ == "__main__":
    main()

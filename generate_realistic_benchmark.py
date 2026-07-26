"""
Generate realistic synthetic face embeddings for benchmarking.

This creates embeddings that mimic real face recognition embeddings:
- Same-person pairs: high cosine similarity, correlated dimensions
- Different-person pairs: low cosine similarity, uncorrelated dimensions
- Includes some noise and outliers to simulate real-world conditions
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imgnet.eval import Sample


def generate_realistic_embeddings(
    n_pairs: int = 50,
    dim: int = 1024,
    same_person_correlation: float = 0.85,
    different_person_correlation: float = 0.05,
    noise_std: float = 0.15,
    seed: int = 42,
) -> list[Sample]:
    """
    Generate synthetic but realistic face embeddings for benchmarking.
    
    Args:
        n_pairs: Number of pairs to generate (half same, half different)
        dim: Embedding dimension
        same_person_correlation: How correlated same-person embeddings are
        different_person_correlation: How correlated different-person embeddings are
        noise_std: Standard deviation of noise added to embeddings
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    samples = []
    
    # Generate base identities
    n_identities = n_pairs // 2 + 10  # Extra identities for negative pairs
    identity_embeddings = rng.standard_normal((n_identities, dim))
    # Normalize each identity embedding
    identity_embeddings = identity_embeddings / np.linalg.norm(identity_embeddings, axis=1, keepdims=True)
    
    for i in range(n_pairs):
        if i % 2 == 0:
            # Same person pair
            identity_idx = i // 2 % n_identities
            base = identity_embeddings[identity_idx]
            # Add correlated noise
            noise1 = rng.normal(0, noise_std, dim)
            noise2 = rng.normal(0, noise_std, dim)
            emb1 = same_person_correlation * base + (1 - same_person_correlation) * noise1
            emb2 = same_person_correlation * base + (1 - same_person_correlation) * noise2
            # Normalize
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
            samples.append(Sample(
                sample_id=f"same_{i:03d}",
                label="same",
                embedding_a=emb1.astype(np.float64),
                embedding_b=emb2.astype(np.float64),
            ))
        else:
            # Different person pair
            idx1, idx2 = rng.choice(n_identities, 2, replace=False)
            base1 = identity_embeddings[idx1]
            base2 = identity_embeddings[idx2]
            noise1 = rng.normal(0, noise_std, dim)
            noise2 = rng.normal(0, noise_std, dim)
            emb1 = different_person_correlation * base1 + (1 - different_person_correlation) * noise1
            emb2 = different_person_correlation * base2 + (1 - different_person_correlation) * noise2
            # Normalize
            emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
            emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
            samples.append(Sample(
                sample_id=f"diff_{i:03d}",
                label="different",
                embedding_a=emb1.astype(np.float64),
                embedding_b=emb2.astype(np.float64),
            ))
    
    return samples


def save_samples(samples: list[Sample], output_dir: Path) -> None:
    """Save samples to NPY files with JSON metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        np.save(output_dir / f"{sample.sample_id}_A.npy", sample.embedding_a)
        np.save(output_dir / f"{sample.sample_id}_B.npy", sample.embedding_b)
        (output_dir / f"{sample.sample_id}.json").write_text(
            json.dumps({"label": sample.label}), encoding="utf-8"
        )
    print(f"Saved {len(samples)} samples to {output_dir}")


if __name__ == "__main__":
    samples = generate_realistic_embeddings(n_pairs=100, seed=42)
    save_samples(samples, Path("benchmark_realistic"))
    
    # Print some stats
    same_cosines = []
    diff_cosines = []
    for s in samples:
        cos = np.dot(s.embedding_a, s.embedding_b)
        if s.label == "same":
            same_cosines.append(cos)
        else:
            diff_cosines.append(cos)
    
    print(f"\nStatistics:")
    print(f"  Same person pairs: {len(same_cosines)}")
    print(f"    Cosine range: [{min(same_cosines):.3f}, {max(same_cosines):.3f}]")
    print(f"    Cosine mean: {np.mean(same_cosines):.3f}")
    print(f"  Different person pairs: {len(diff_cosines)}")
    print(f"    Cosine range: [{min(diff_cosines):.3f}, {max(diff_cosines):.3f}]")
    print(f"    Cosine mean: {np.mean(diff_cosines):.3f}")

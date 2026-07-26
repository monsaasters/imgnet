"""
Tests for imgnet.benchmark
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imgnet.benchmark import (
    BenchmarkConfig,
    EmbeddingPair,
    load_pairs_from_csv,
    load_pairs_from_npy_dir,
    run_benchmark,
    summarize,
    to_csv,
    to_json,
)


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(1024)
    return EmbeddingPair(
        pair_id="p",
        label="same",
        embedding_a=base,
        embedding_b=base + rng.normal(0, 0.1, 1024),
    )


class TestRunBenchmark:
    def test_result_count(self):
        cfg = BenchmarkConfig(severities=[0.1], strategies=["random_noise"], seed=1)
        results = run_benchmark([_pair(0)], cfg)
        assert len(results) == 1

    def test_delta_sign(self):
        cfg = BenchmarkConfig(severities=[0.5], strategies=["block_zero"], seed=1)
        results = run_benchmark([_pair(0)], cfg)
        assert all(r.img_sign_delta >= 0 for r in results)

    def test_votes_present(self):
        cfg = BenchmarkConfig(severities=[0.2], strategies=["random_noise"], seed=1)
        results = run_benchmark([_pair(0)], cfg)
        votes = {r.vote for r in results}
        assert votes.issubset({"MATCH", "UNCERTAIN", "DIFFERENT"})


class TestSummarize:
    def test_keys(self):
        cfg = BenchmarkConfig(severities=[0.1], strategies=["random_noise"], seed=1)
        results = run_benchmark([_pair(0)], cfg)
        s = summarize(results)
        assert "avg_img_sign" in s
        assert "avg_cosine" in s
        assert "by_strategy" in s


class TestIo:
    def test_json_roundtrip(self, tmp_path):
        cfg = BenchmarkConfig(severities=[0.1], strategies=["random_noise"], seed=1)
        results = run_benchmark([_pair(0)], cfg)
        out = tmp_path / "out.json"
        to_json(results, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total_results"] == len(results)

    def test_csv_roundtrip(self, tmp_path):
        cfg = BenchmarkConfig(severities=[0.1], strategies=["random_noise"], seed=1)
        results = run_benchmark([_pair(0)], cfg)
        out = tmp_path / "out.csv"
        to_csv(results, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("pair_id,")
        assert len(lines) == len(results) + 1

    def test_load_npy_dir(self, tmp_path):
        for i in range(3):
            np.save(tmp_path / f"pair_{i}_A.npy", np.random.randn(1024))
            np.save(tmp_path / f"pair_{i}_B.npy", np.random.randn(1024))
        pairs = load_pairs_from_npy_dir(tmp_path)
        assert len(pairs) == 3

    def test_load_csv(self, tmp_path):
        rows = np.random.randn(4, 1024)
        np.savetxt(tmp_path / "pairs.csv", rows, delimiter=",")
        pairs = load_pairs_from_csv(tmp_path / "pairs.csv", embedding_cols=range(1024))
        assert len(pairs) == 2

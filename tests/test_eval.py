"""
Tests for imgnet.eval
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from imgnet.eval import EvalConfig, Sample, evaluate_samples, load_samples_from_csv, load_samples_from_npy_dir, results_to_csv, results_to_json, summarize_eval


def _make_samples(n=10, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        base = rng.standard_normal(1024)
        label = "same" if i % 2 == 0 else "different"
        if label == "same":
            b = base + rng.normal(0, 0.05, 1024)
        else:
            b = rng.standard_normal(1024)
        samples.append(Sample(sample_id=f"s{i}", label=label, embedding_a=base, embedding_b=b))
    return samples


class TestEvaluateSamples:
    def test_output_length(self):
        samples = _make_samples(6)
        results = evaluate_samples(samples, EvalConfig())
        assert len(results) == 6

    def test_vote_labels(self):
        samples = _make_samples(6)
        results = evaluate_samples(samples, EvalConfig())
        for r in results:
            assert r.vote in {"MATCH", "UNCERTAIN", "DIFFERENT"}

    def test_correctness_same(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal(1024)
        samples = [Sample("same", "same", base, base + rng.normal(0, 0.01, 1024))]
        results = evaluate_samples(samples, EvalConfig())
        assert results[0].correct is True

    def test_correctness_different(self):
        rng = np.random.default_rng(1)
        base = rng.standard_normal(1024)
        samples = [Sample("diff", "different", base, rng.standard_normal(1024))]
        results = evaluate_samples(samples, EvalConfig())
        assert results[0].correct is True


class TestSummarize:
    def test_keys(self):
        samples = _make_samples(10)
        results = evaluate_samples(samples, EvalConfig())
        s = summarize_eval(results)
        assert "accuracy" in s
        assert "vote_counts" in s


class TestIo:
    def test_json_roundtrip(self, tmp_path):
        samples = _make_samples(5)
        results = evaluate_samples(samples, EvalConfig())
        out = tmp_path / "out.json"
        results_to_json(results, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["summary"]["total"] == len(results)

    def test_csv_roundtrip(self, tmp_path):
        samples = _make_samples(5)
        results = evaluate_samples(samples, EvalConfig())
        out = tmp_path / "out.csv"
        results_to_csv(results, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("sample_id,")
        assert len(lines) == len(results) + 1

    def test_load_npy_dir(self, tmp_path):
        for i in range(4):
            np.save(tmp_path / f"pair_{i}_A.npy", np.random.randn(1024))
            np.save(tmp_path / f"pair_{i}_B.npy", np.random.randn(1024))
        samples = load_samples_from_npy_dir(tmp_path)
        assert len(samples) == 4

    def test_load_csv(self, tmp_path):
        rows = np.random.randn(4, 1024)
        np.savetxt(tmp_path / "pairs.csv", rows, delimiter=",")
        samples = load_samples_from_csv(tmp_path / "pairs.csv", embedding_cols=range(1024))
        assert len(samples) == 2

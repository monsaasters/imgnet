"""
Tests for imgnet.metrics
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imgnet.metrics import (
    amp_img_score,
    chain_score,
    cosine_similarity,
    img_sign_score,
    batch_compare,
)


def _make_pair(seed=0, dim=1024):
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(dim)
    e2 = e1 + rng.normal(0, 0.1, dim)
    return e1.tolist(), e2.tolist()


class TestImgSignScore:
    def test_identical_embeddings(self):
        e = [0.1, -0.5, 0.3, -0.2, 0.8]
        score = img_sign_score(e, e, window_size=3, threshold=2)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_opposite_embeddings(self):
        e1 = [0.1, -0.5, 0.3, -0.2, 0.8]
        e2 = [-x for x in e1]
        score = img_sign_score(e1, e2, window_size=3, threshold=2)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_score_range(self):
        e1, e2 = _make_pair(dim=1024)
        score = img_sign_score(e1, e2)
        assert 0.0 <= score <= 1.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            img_sign_score([1, 2], [1, 2, 3])


class TestAmpImgScore:
    def test_range(self):
        e1, e2 = _make_pair(dim=1024)
        score = amp_img_score(e1, e2)
        assert 0.0 <= score <= 1.0


class TestChainScore:
    def test_range(self):
        e1, e2 = _make_pair(dim=1024)
        score, chains, avg_chain = chain_score(e1, e2)
        assert 0.0 <= score <= 1.0
        assert chains >= 0
        assert avg_chain >= 0.0


class TestCosineSimilarity:
    def test_identical(self):
        e = [0.1, -0.5, 0.3]
        assert cosine_similarity(e, e) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_opposite(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0, abs=1e-6)


class TestBatchCompare:
    def test_output_keys(self):
        e1, e2 = _make_pair(dim=1024)
        result = batch_compare(e1, e2)
        assert "img_sign" in result
        assert "amp_img" in result
        assert "chain_score" in result
        assert "cosine" in result
        assert "vote" in result
        assert result["vote"] in ("MATCH", "UNCERTAIN", "DIFFERENT")

    def test_same_embedding_is_match(self):
        e = [0.1, -0.5, 0.3, -0.2] * 256
        result = batch_compare(e, e)
        assert result["vote"] == "MATCH"
        assert result["img_sign"] == pytest.approx(1.0, abs=1e-6)
        assert result["cosine"] == pytest.approx(1.0, abs=1e-6)

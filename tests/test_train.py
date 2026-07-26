"""
Tests for imgnet.train
"""

from __future__ import annotations

import sys

import pytest

# If torch is not installed, skip these tests.
torch = pytest.importorskip("torch", reason="torch is required for training tests")

from imgnet.train import IMGNet, TrainConfig, contrastive_loss, img_sign_score_torch  # noqa: E402

try:
    from imgnet.train import build_trainer  # noqa: F401
    BUILD_TRAINER_OK = True
except Exception:
    BUILD_TRAINER_OK = False


class TestModel:
    def test_forward_shape(self):
        model = IMGNet(emb_dim=128)
        x = torch.zeros(1, 3, 112, 112)
        out = model(x)
        assert out.shape == (1, 128)

    def test_param_count(self):
        model = IMGNet(emb_dim=1024)
        assert model.n_params() > 0


class TestLoss:
    def test_contrastive_loss(self):
        E1 = torch.randn(4, 128)
        E2 = torch.randn(4, 128)
        labels = torch.tensor([1, 1, 0, 0])
        sm = labels == 1
        dm = labels == 0
        loss, ls, ld = contrastive_loss(E1[sm], E2[sm], E1[dm], E2[dm], TrainConfig(emb_dim=128))
        assert ls >= 0.0
        assert ld >= 0.0
        assert float(loss) >= 0.0

    def test_img_sign_score_range(self):
        E1 = torch.randn(4, 1024)
        E2 = torch.randn(4, 1024)
        score = img_sign_score_torch(E1, E2)
        assert score.shape == (4,)
        assert torch.all((score >= 0.0) & (score <= 1.0))

    def test_identical_embeddings_high_score(self):
        E = torch.randn(2, 1024)
        score = img_sign_score_torch(E, E)
        assert float(score.mean()) > 0.9


class TestTrainerBuild:
    @pytest.mark.skipif(not BUILD_TRAINER_OK, reason="trainer build requires full training deps")
    def test_build_requires_existing_data(self, tmp_path):
        cfg = TrainConfig(data_root=str(tmp_path), ckpt_root=str(tmp_path / "ckpts"), num_epochs=1)
        with pytest.raises(Exception):
            build_trainer(cfg)

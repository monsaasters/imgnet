"""
IMGNet — Universal Similarity Metrics Library
=============================================

IMG is proposed as an *alternative* similarity metric, not a replacement for
cosine similarity. The optimal metric depends on how the embedding itself is
learned.

This package exposes:
- img_sign(e1, e2)
- amp_img(e1, e2)
- chain_score(e1, e2)
- cosine_similarity(e1, e2)
- batch_compare(embeddings, metric="img_sign")
- IMGNet model + training utilities

and CLI / API wrappers for verification use-cases.

Author: Imam Ghozali
License: MIT
"""

from .metrics import (
    img_sign_score,
    amp_img_score,
    chain_score as chain_score_fn,
    cosine_similarity,
    batch_compare,
)

# Training utilities are optional; they require torch.
try:
    from .train import (
        IMGNet,
        TrainConfig,
        build_trainer,
        fit,
        contrastive_loss,
        img_sign_score_torch,
    )
    _TRAIN_OK = True
except Exception:
    _TRAIN_OK = False

__all__ = [
    "img_sign_score",
    "amp_img_score",
    "chain_score_fn",
    "cosine_similarity",
    "batch_compare",
]

if _TRAIN_OK:
    __all__ += [
        "IMGNet",
        "TrainConfig",
        "build_trainer",
        "fit",
        "contrastive_loss",
        "img_sign_score_torch",
    ]

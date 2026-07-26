"""
ImgNet visualizer utilities

Provides image-to-embedding helpers used by the demo app.
If the IMGNet checkpoint exists and torch is available, it uses the real model.
Otherwise it falls back to a deterministic content-based embedding so the API
and demo remain functional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from facenet_pytorch import MTCNN

    _mtcnn = MTCNN(image_size=112, keep_all=False, post_process=False, device="cpu")
    MTCNN_OK = True
except Exception:
    _mtcnn = None
    MTCNN_OK = False


def _default_checkpoint_paths() -> list[Path]:
    """Return candidate checkpoint paths in cross-platform order."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "best_model_epoch39_plateau.pth",
        here.parent.parent / "best_model_epoch39_plateau.pth",
        Path.cwd() / "best_model_epoch39_plateau.pth",
        Path.home() / ".imgnet" / "best_model_epoch39_plateau.pth",
    ]
    # Preserve existing Windows-style path if it exists
    legacy = Path(r"C:\PythonProj\img_bnn\checkpoints_sw357_conv10_imgsign\SW357_conv10_imgsign\best_model_epoch39_plateau.pth")
    if legacy.exists():
        candidates.insert(0, legacy)
    return candidates


_CKPT_PATH = None
for p in _default_checkpoint_paths():
    if p.exists():
        _CKPT_PATH = p
        break

if TORCH_OK:
    class _SWBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(240, 64), nn.ReLU(True), nn.Linear(64, 32))

        def forward(self, x):
            diffs = []
            for ws in (3, 5, 7):
                p = ws // 2
                x_pad = F.pad(x, [p, p, p, p], mode="reflect")
                patches = x_pad.unfold(2, ws, 1).unfold(3, ws, 1)
                diff = x.unsqueeze(-1).unsqueeze(-1) - patches
                mid = ws // 2
                mask = torch.ones(ws, ws, dtype=torch.bool, device=x.device)
                mask[mid, mid] = False
                diffs.append(diff[:, :, :, :, mask])
            d = torch.cat(diffs, -1)
            B, C, H, W, N = d.shape
            out = self.fc(d.permute(0, 2, 3, 1, 4).reshape(B * H * W, C * N))
            return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

    class _IMGNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sw1 = _SWBlock()
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(128)
            self.conv5 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False)
            self.bn6 = nn.BatchNorm2d(128)
            self.conv7 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
            self.bn7 = nn.BatchNorm2d(256)
            self.conv8 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.bn8 = nn.BatchNorm2d(256)
            self.conv9 = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)
            self.bn9 = nn.BatchNorm2d(256)
            self.conv10 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
            self.bn10 = nn.BatchNorm2d(256)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 1024)
            self.bn = nn.BatchNorm1d(1024)

        def forward(self, x):
            x = F.relu(self.bn1(self.sw1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = F.relu(self.bn7(self.conv7(x)))
            x = F.relu(self.bn8(self.conv8(x)))
            x = F.relu(self.bn9(self.conv9(x)))
            x = F.relu(self.bn10(self.conv10(x)))
            x = self.gap(x).view(x.size(0), -1)
            return self.bn(self.fc(x))

    _model: Optional[nn.Module] = None
    _model_error: Optional[str] = None
    if _CKPT_PATH is not None:
        try:
            _model = _IMGNet()
            state = torch.load(str(_CKPT_PATH), map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            _model.load_state_dict(state, strict=True)
            _model.eval()
        except Exception as exc:
            _model = None
            _model_error = str(exc)
else:
    _model = None
    _model_error = "PyTorch is not installed."


def _dummy_embed(img_array: np.ndarray) -> np.ndarray:
    flat = img_array.flatten().astype(np.float32) / 255.0
    np.random.seed(int(flat.sum() * 1000) % (2**31))
    emb = np.random.randn(1024).astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)


def _content_embed(img_array: np.ndarray) -> np.ndarray:
    img = img_array.astype(np.float32) / 255.0
    if img.ndim == 3:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        gray = img
    gray = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    emb = np.fft.rfft2(gray).real.flatten()
    emb = np.pad(emb, (0, max(0, 1024 - emb.shape[0])), mode="wrap")
    emb = emb[:1024]
    return (emb - emb.mean()) / (emb.std() + 1e-6)


def _crop_face(img_rgb: np.ndarray) -> np.ndarray:
    """Detect and crop face to 112x112. If detection fails, resize instead."""
    if MTCNN_OK and _mtcnn is not None:
        try:
            pil = Image.fromarray(img_rgb)
            face = _mtcnn(pil)
            if face is not None:
                arr = face.permute(1, 2, 0).cpu().numpy()
                return np.clip(arr, 0, 255).astype(np.uint8)
        except Exception:
            pass
    return np.array(Image.fromarray(img_rgb).resize((112, 112), Image.Resampling.BILINEAR), dtype=np.uint8)


def compute_embedding(img_array: np.ndarray, *, use_real_model: bool = True) -> np.ndarray:
    if use_real_model and TORCH_OK and _model is not None:
        try:
            cropped = _crop_face(img_array)
            t = torch.from_numpy(cropped.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                emb = _model(t).squeeze(0).cpu().numpy()
            return emb
        except Exception:
            pass
    return _content_embed(img_array)


def model_status() -> dict:
    return {
        "torch": bool(TORCH_OK),
        "mtcnn": bool(MTCNN_OK),
        "checkpoint": str(_CKPT_PATH) if _CKPT_PATH else None,
        "model_loaded": bool(_model is not None),
        "error": _model_error,
    }

"""
IMGNet training utilities

Modular training pipeline:
- Model definition
- IMG metrics loss
- Dataset / dataloader
- Trainer with checkpoint/resume
- Hybrid cosine+IMG mode
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from PIL import Image
    import torchvision.transforms as T

    PIL_OK = True
except Exception:
    PIL_OK = False


# ============================================================
# CONFIG
# ============================================================

@dataclass
class TrainConfig:
    data_root: str = "./data"
    ckpt_root: str = "./checkpoints"
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_epochs: int = 5
    window_size: int = 11
    threshold: int = 8
    emb_dim: int = 1024
    max_pairs_per_identity: int = 300
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    name: str = "SW357_conv10_imgsign"
    resume: bool = True
    hybrid_mode: bool = False
    hybrid_cosine_weight: float = 0.5


# ============================================================
# MODEL
# ============================================================

class SWBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, window_sizes=(3, 5, 7)):
        super().__init__()
        self.window_sizes = window_sizes
        n_diff = sum(w * w - 1 for w in window_sizes)
        n_input = n_diff * in_ch
        self.fc = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_ch),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        diffs = []
        for ws in self.window_sizes:
            pad = ws // 2
            x_pad = F.pad(x, [pad, pad, pad, pad], mode="reflect")
            patches = x_pad.unfold(2, ws, 1).unfold(3, ws, 1)
            center = x.unsqueeze(-1).unsqueeze(-1)
            diff = center - patches
            mid = ws // 2
            mask = torch.ones(ws, ws, dtype=torch.bool, device=x.device)
            mask[mid, mid] = False
            diff = diff[:, :, :, :, mask]
            diffs.append(diff)
        diffs = torch.cat(diffs, dim=-1)
        B, C, H, W, N = diffs.shape
        diffs = diffs.permute(0, 2, 3, 1, 4).reshape(B * H * W, C * N)
        out = self.fc(diffs)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class IMGNet(nn.Module):
    def __init__(self, emb_dim: int = 1024):
        super().__init__()
        self.sw1 = SWBlock(3, 32, window_sizes=[3, 5, 7])
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim)

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

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# METRICS / LOSS
# ============================================================

def img_sign_score_torch(
    E1: torch.Tensor,
    E2: torch.Tensor,
    beta: float = 10.0,
    window_size: int = 11,
    threshold: int = 8,
) -> torch.Tensor:
    kernel = torch.ones(1, 1, window_size, device=E1.device)
    agreement = (torch.tanh(beta * E1 * E2) + 1) / 2
    soft_match = F.conv1d(agreement.unsqueeze(1), kernel, stride=1).squeeze(1)
    gate = torch.sigmoid(50.0 * (soft_match - threshold + 0.5))
    return gate.mean(dim=1)


def cosine_similarity_torch(E1: torch.Tensor, E2: torch.Tensor) -> torch.Tensor:
    return (E1 * E2).sum(dim=1) / (torch.norm(E1, dim=1) * torch.norm(E2, dim=1) + 1e-8)


def contrastive_loss(
    E1_s: torch.Tensor,
    E2_s: torch.Tensor,
    E1_d: torch.Tensor,
    E2_d: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, float, float]:
    device = E1_s.device if E1_s.shape[0] > 0 else E2_d.device
    ls = torch.tensor(0.0, device=device)
    ld = torch.tensor(0.0, device=device)
    ls_item = 0.0
    ld_item = 0.0

    if E1_s.shape[0] > 0:
        img_s = img_sign_score_torch(E1_s, E2_s, window_size=cfg.window_size, threshold=cfg.threshold)
        cos_s = cosine_similarity_torch(E1_s, E2_s)
        if cfg.hybrid_mode:
            ls = ((1.0 - cfg.hybrid_cosine_weight) * (1.0 - img_s) ** 2 + cfg.hybrid_cosine_weight * (1.0 - cos_s) ** 2).mean()
        else:
            ls = ((1.0 - img_s) ** 2).mean()
        ls_item = ls.item()

    if E2_d.shape[0] > 0:
        img_d = img_sign_score_torch(E1_d, E2_d, window_size=cfg.window_size, threshold=cfg.threshold)
        cos_d = cosine_similarity_torch(E1_d, E2_d)
        if cfg.hybrid_mode:
            ld = ((1.0 - cfg.hybrid_cosine_weight) * (img_d) ** 2 + cfg.hybrid_cosine_weight * (cos_d) ** 2).mean()
        else:
            ld = (img_d ** 2).mean()
        ld_item = ld.item()

    return ls + ld, ls_item, ld_item


# ============================================================
# DATASET
# ============================================================

class PairDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 112, max_pairs_per_identity: int = 300, augment: bool = False):
        if not PIL_OK:
            raise RuntimeError("PIL/torchvision is required for training.")
        self.img_size = img_size
        self.augment = augment
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data root not found: {root_dir}")

        self.identity_images: dict[str, list[str]] = {}
        identities = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        for idx, identity in enumerate(identities):
            path = self.root_dir / identity
            images = [str(p) for p in path.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}]
            if len(images) >= 2:
                self.identity_images[identity] = images
            if (idx + 1) % 1000 == 0:
                print(f"  scanning... {idx+1}/{len(identities)}")

        self.identity_list = list(self.identity_images.keys())
        self.pos_pairs: list[Tuple[str, str]] = []
        for identity, images in self.identity_images.items():
            n = min(max_pairs_per_identity, len(images))
            for _ in range(n):
                i, j = random.sample(range(len(images)), 2)
                self.pos_pairs.append((images[i], images[j]))
        self.n_neg = len(self.pos_pairs)

    def __len__(self) -> int:
        return len(self.pos_pairs) + self.n_neg

    def _load(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        if self.augment:
            aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                T.RandomRotation(degrees=10),
                T.RandomGrayscale(p=0.1),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
            t = aug(t)
        return t

    def _random_negative(self) -> Tuple[str, str]:
        id1, id2 = random.sample(self.identity_list, 2)
        return random.choice(self.identity_images[id1]), random.choice(self.identity_images[id2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < len(self.pos_pairs):
            p1, p2 = self.pos_pairs[idx]
            return self._load(p1), self._load(p2), torch.tensor(1)
        p1, p2 = self._random_negative()
        return self._load(p1), self._load(p2), torch.tensor(0)


# ============================================================
# TRAINER
# ============================================================

@dataclass
class Trainer:
    model: nn.Module
    cfg: TrainConfig
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    optimizer: Optional[torch.optim.Optimizer] = field(default=None)
    start_epoch: int = 0
    best_val: float = field(default=float("inf"))


def build_trainer(cfg: TrainConfig) -> Trainer:
    device = torch.device(cfg.device)
    model = IMGNet(emb_dim=cfg.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_dataset = PairDataset(cfg.data_root, max_pairs_per_identity=cfg.max_pairs_per_identity, augment=True)
    val_dataset = PairDataset(cfg.data_root, max_pairs_per_identity=cfg.max_pairs_per_identity, augment=False)

    total = len(train_dataset)
    indices = list(range(total))
    random.seed(cfg.seed)
    random.shuffle(indices)
    val_size = int(total * 0.1)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    pin = device.type == "cuda"
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )

    start_epoch = 0
    best_val = float("inf")
    ckpt_dir = Path(cfg.ckpt_root) / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / "last_checkpoint.pth"

    if cfg.resume and resume_path.exists():
        try:
            ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", float("inf"))
            print(f"  Resumed from epoch {start_epoch}")
        except RuntimeError:
            print("  Checkpoint incompatible, training from scratch")

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Batches/epoch: {len(train_loader)}")

    return Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        start_epoch=start_epoch,
        best_val=best_val,
    )


def train_epoch(trainer: Trainer, epoch: int) -> Tuple[float, float, float]:
    model = trainer.model
    cfg = trainer.cfg
    device = torch.device(cfg.device)
    model.train()
    t_loss = t_s = t_d = 0.0
    n = 0
    for batch_idx, (img1, img2, labels) in enumerate(trainer.train_loader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        trainer.optimizer.zero_grad()
        E1, E2 = model(img1), model(img2)
        sm, dm = labels == 1, labels == 0
        loss, ls, ld = contrastive_loss(E1[sm], E2[sm], E1[dm], E2[dm], cfg)
        if loss.item() > 0:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
        t_loss += loss.item()
        t_s += ls
        t_d += ld
        n += 1
        if batch_idx == 0:
            print(f"  Epoch {epoch+1} started...")
        if (batch_idx + 1) % 100 == 0:
            with torch.no_grad():
                s_mean = img_sign_score_torch(E1[sm], E2[sm]).mean().item() if sm.sum() > 0 else 0.0
                d_mean = img_sign_score_torch(E1[dm], E2[dm]).mean().item() if dm.sum() > 0 else 0.0
            print(f"  batch {batch_idx+1}/{len(trainer.train_loader)} loss={loss.item():.4f} | same={s_mean:.3f} diff={d_mean:.3f}")
    return t_loss / max(n, 1), t_s / max(n, 1), t_d / max(n, 1)


def validate(trainer: Trainer) -> float:
    model = trainer.model
    cfg = trainer.cfg
    device = torch.device(cfg.device)
    model.eval()
    v_loss = 0.0
    nv = 0
    with torch.no_grad():
        for img1, img2, labels in trainer.val_loader:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            E1, E2 = model(img1), model(img2)
            sm, dm = labels == 1, labels == 0
            loss, _, _ = contrastive_loss(E1[sm], E2[sm], E1[dm], E2[dm], cfg)
            v_loss += loss.item()
            nv += 1
    return v_loss / max(nv, 1)


def fit(trainer: Trainer) -> nn.Module:
    cfg = trainer.cfg
    device = torch.device(cfg.device)
    ckpt_dir = Path(cfg.ckpt_root) / cfg.name

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lambda ep: (ep + 1) / cfg.warmup_epochs if ep < cfg.warmup_epochs else 1.0,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=cfg.num_epochs - cfg.warmup_epochs,
        eta_min=1e-6,
    )

    for epoch in range(trainer.start_epoch, cfg.num_epochs):
        train_loss, train_s, train_d = train_epoch(trainer, epoch)
        val_loss = validate(trainer)

        if epoch < cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{cfg.num_epochs} | "
            f"Train {train_loss:.4f} (same={train_s:.4f} diff={train_d:.4f}) | "
            f"Val {val_loss:.4f} | LR {current_lr:.6f}"
        )

        if val_loss < trainer.best_val:
            trainer.best_val = val_loss
            best_path = ckpt_dir / f"best_model_epoch{epoch+1}.pth"
            torch.save(trainer.model.state_dict(), best_path)
            print(f"  -> best saved: best_model_epoch{epoch+1}.pth (val={val_loss:.4f})")

        torch.save(
            {
                "epoch": epoch,
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val": trainer.best_val,
                "config": cfg.__dict__,
            },
            ckpt_dir / "last_checkpoint.pth",
        )

    final_path = ckpt_dir / "final_model.pth"
    torch.save(trainer.model.state_dict(), final_path)
    print(f"Training complete! Final saved: {final_path}")
    return trainer.model

"""Patient-independent WGAN-GP ictal generator (plan Section 8, Provider A).

A reproducible, deployment-relevant GAN baseline: trained on pooled training-
patient ictal windows, it generates generic ictal windows for patients it has
never seen. Stable WGAN-GP config (latent 128, critic 5x, gradient penalty 10,
linear output with clip). Flagged/skipped if fewer than ``min_ictal_windows``
real ictal windows are available in the subset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from synthetic.provider_base import SyntheticProvider, ictal_subset


@dataclass
class WGANConfig:
    latent_dim: int = 128
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-4
    n_critic: int = 5
    gp_lambda: float = 10.0
    clip_value: float = 6.0
    min_ictal_windows: int = 256
    base_time_divisor: int = 16
    device: str = "cpu"
    seed: int = 42


def _fix_length(x: torch.Tensor, target: int) -> torch.Tensor:
    """Crop or zero-pad the time axis of ``(N, C, T)`` to ``target``."""
    T = x.shape[-1]
    if T == target:
        return x
    if T > target:
        start = (T - target) // 2
        return x[..., start:start + target]
    pad = target - T
    return torch.nn.functional.pad(x, (pad // 2, pad - pad // 2))


class _Generator(nn.Module):
    def __init__(self, latent_dim, n_channels, t0, base=128):
        super().__init__()
        self.t0, self.base = t0, base
        self.project = nn.Linear(latent_dim, base * t0)
        self.net = nn.Sequential(
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.ConvTranspose1d(base, 64, 4, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, n_channels, 4, stride=2, padding=1),  # linear output
        )

    def forward(self, z):
        h = self.project(z).view(z.shape[0], self.base, self.t0)
        return self.net(h)


class _Critic(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # No batchnorm in the critic (WGAN-GP); LeakyReLU stack.
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class WGANGPProvider(SyntheticProvider):
    name = "wgan_gp"
    paradigm = "patient_independent"

    def __init__(self, config: Optional[WGANConfig] = None):
        super().__init__()
        self.cfg = config or WGANConfig()
        self.skipped = False
        self.n_channels = None
        self.n_samples = None
        self.G = None

    def _gradient_penalty(self, critic, real, fake, device):
        n = real.shape[0]
        eps = torch.rand(n, 1, 1, device=device)
        inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d = critic(inter)
        grads = torch.autograd.grad(
            outputs=d, inputs=inter, grad_outputs=torch.ones_like(d),
            create_graph=True, retain_graph=True)[0]
        grads = grads.reshape(n, -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, train_windows, train_labels, train_metadata, config=None):
        cfg = self.cfg
        device = cfg.device
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        X = ictal_subset(train_windows, train_labels)
        self.metadata = self._build_metadata(train_metadata, n_ictal=len(X))
        self.metadata.extra["config"] = cfg.__dict__.copy()
        if len(X) < cfg.min_ictal_windows:
            self.skipped = True
            self.fitted = False
            self.metadata.extra["skipped"] = True
            self.metadata.extra["skip_reason"] = (
                f"only {len(X)} ictal windows < min {cfg.min_ictal_windows}")
            return self

        self.n_channels, self.n_samples = X.shape[1], X.shape[2]
        t0 = max(1, self.n_samples // cfg.base_time_divisor)
        self.G = _Generator(cfg.latent_dim, self.n_channels, t0).to(device)
        critic = _Critic(self.n_channels).to(device)
        optG = torch.optim.Adam(self.G.parameters(), lr=cfg.lr, betas=(0.0, 0.9))
        optC = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(0.0, 0.9))

        data = torch.from_numpy(X).float()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=cfg.batch_size,
            shuffle=True, drop_last=True)
        if len(loader) == 0:  # batch larger than dataset
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(data), batch_size=len(data), shuffle=True)

        for epoch in range(cfg.epochs):
            for (real,) in loader:
                real = real.to(device)
                n = real.shape[0]
                for _ in range(cfg.n_critic):
                    z = torch.randn(n, cfg.latent_dim, device=device)
                    fake = _fix_length(self.G(z), self.n_samples).detach()
                    optC.zero_grad()
                    loss_c = (critic(fake).mean() - critic(real).mean()
                              + cfg.gp_lambda * self._gradient_penalty(critic, real, fake, device))
                    loss_c.backward()
                    optC.step()
                z = torch.randn(n, cfg.latent_dim, device=device)
                fake = _fix_length(self.G(z), self.n_samples)
                optG.zero_grad()
                (-critic(fake).mean()).backward()
                optG.step()

        self.fitted = True
        return self

    def generate(self, n: int, class_label: str = "seizure", seed: int = 42) -> np.ndarray:
        if self.skipped or not self.fitted or self.G is None:
            raise RuntimeError("WGAN-GP provider was not fitted (skipped or below threshold)")
        torch.manual_seed(seed)
        self.G.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, 256):
                k = min(256, n - i)
                z = torch.randn(k, self.cfg.latent_dim, device=self.cfg.device)
                x = _fix_length(self.G(z), self.n_samples)
                out.append(x.cpu().numpy())
        x = np.concatenate(out).astype("float32")
        if self.cfg.clip_value:
            x = np.clip(x, -self.cfg.clip_value, self.cfg.clip_value)
        return x

    def _save_state(self, path: Path) -> None:
        if self.G is not None:
            torch.save({"state_dict": self.G.state_dict(),
                        "n_channels": self.n_channels, "n_samples": self.n_samples,
                        "config": self.cfg.__dict__}, path / "generator.pt")

    def _load_state(self, path: Path) -> None:
        ckpt_path = path / "generator.pt"
        if not ckpt_path.exists():
            self.skipped = True
            return
        ckpt = torch.load(ckpt_path, map_location=self.cfg.device, weights_only=False)
        self.n_channels, self.n_samples = ckpt["n_channels"], ckpt["n_samples"]
        t0 = max(1, self.n_samples // self.cfg.base_time_divisor)
        self.G = _Generator(self.cfg.latent_dim, self.n_channels, t0).to(self.cfg.device)
        self.G.load_state_dict(ckpt["state_dict"])

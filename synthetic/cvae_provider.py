"""Patient-independent conditional VAE ictal generator (plan Section 8, Provider B).

The strong generator we control end-to-end (no external release dependency). A
conditional VAE over ``(C, T)`` ictal windows: a temporal-conv encoder produces
``N(mu, sigma)``; a conv up-sampling decoder reconstructs windows. Trained
patient-independently on training ictal windows. Contrasts with the GAN's
failure modes and aligns with the VAE/LDM components of recent strong EEG
generators.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic.provider_base import SyntheticProvider, ictal_subset
from synthetic.wgan_gp_provider import _fix_length


@dataclass
class CVAEConfig:
    latent_dim: int = 64
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    beta: float = 1.0
    num_classes: int = 1
    min_ictal_windows: int = 256
    base_time_divisor: int = 16
    device: str = "cpu"
    seed: int = 42


class _Encoder(nn.Module):
    def __init__(self, n_channels, t0, latent_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_channels, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 4, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 4, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.flat = 128 * t0
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_lv = nn.Linear(self.flat, latent_dim)

    def forward(self, x):
        h = self.body(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)


class _Decoder(nn.Module):
    def __init__(self, n_channels, t0, latent_dim, num_classes):
        super().__init__()
        self.t0 = t0
        self.cls = nn.Embedding(max(1, num_classes), latent_dim)
        self.project = nn.Linear(latent_dim, 128 * t0)
        self.net = nn.Sequential(
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 4, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, n_channels, 4, stride=2, padding=1),  # linear output
        )

    def forward(self, z, cls_idx):
        z = z + self.cls(cls_idx)
        h = self.project(z).view(z.shape[0], 128, self.t0)
        return self.net(h)


class CVAEProvider(SyntheticProvider):
    name = "cvae"
    paradigm = "patient_independent"

    def __init__(self, config: Optional[CVAEConfig] = None):
        super().__init__()
        self.cfg = config or CVAEConfig()
        self.skipped = False
        self.enc = None
        self.dec = None
        self.n_channels = None
        self.n_samples = None
        self.t0 = None

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
        self.t0 = max(1, self.n_samples // cfg.base_time_divisor)
        self.enc = _Encoder(self.n_channels, self.t0, cfg.latent_dim).to(device)
        self.dec = _Decoder(self.n_channels, self.t0, cfg.latent_dim, cfg.num_classes).to(device)
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=cfg.lr)

        data = torch.from_numpy(X).float()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data), batch_size=min(cfg.batch_size, len(data)),
            shuffle=True, drop_last=False)

        for epoch in range(cfg.epochs):
            for (real,) in loader:
                real = real.to(device)
                n = real.shape[0]
                mu, logvar = self.enc(real)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                cls_idx = torch.zeros(n, dtype=torch.long, device=device)
                recon = _fix_length(self.dec(z, cls_idx), self.n_samples)
                # Both terms must be PER-SAMPLE sums for beta to mean what it says. The old
                # code averaged reconstruction over N*C*T = N*18*1024 elements but the KL over
                # N*latent_dim = N*64, over-weighting KL by 18432/64 = 288x -- so a nominal
                # beta = 1.0 behaved like beta ~ 288 and the model posterior-collapsed. With
                # this form beta = 1.0 is the true ELBO.
                # Note: the reconstruction term is now ~18432x larger in absolute value. Adam
                # is approximately invariant to a global loss rescaling, but if the first
                # epochs diverge, lower cfg.lr rather than reverting this.
                recon_loss = F.mse_loss(recon, real, reduction="sum") / n
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
                loss = recon_loss + cfg.beta * kl
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.fitted = True
        return self

    def generate(self, n: int, class_label: str = "seizure", seed: int = 42) -> np.ndarray:
        if self.skipped or not self.fitted or self.dec is None:
            raise RuntimeError("cVAE provider was not fitted (skipped or below threshold)")
        torch.manual_seed(seed)
        self.dec.eval()
        out = []
        with torch.no_grad():
            for i in range(0, n, 256):
                k = min(256, n - i)
                z = torch.randn(k, self.cfg.latent_dim, device=self.cfg.device)
                cls_idx = torch.zeros(k, dtype=torch.long, device=self.cfg.device)
                x = _fix_length(self.dec(z, cls_idx), self.n_samples)
                out.append(x.cpu().numpy())
        return np.concatenate(out).astype("float32")

    def _save_state(self, path: Path) -> None:
        if self.dec is not None:
            torch.save({"enc": self.enc.state_dict(), "dec": self.dec.state_dict(),
                        "n_channels": self.n_channels, "n_samples": self.n_samples,
                        "t0": self.t0, "config": self.cfg.__dict__}, path / "cvae.pt")

    def _load_state(self, path: Path) -> None:
        ckpt_path = path / "cvae.pt"
        if not ckpt_path.exists():
            self.skipped = True
            return
        ckpt = torch.load(ckpt_path, map_location=self.cfg.device, weights_only=False)
        self.n_channels, self.n_samples, self.t0 = ckpt["n_channels"], ckpt["n_samples"], ckpt["t0"]
        self.enc = _Encoder(self.n_channels, self.t0, self.cfg.latent_dim).to(self.cfg.device)
        self.dec = _Decoder(self.n_channels, self.t0, self.cfg.latent_dim,
                            self.cfg.num_classes).to(self.cfg.device)
        self.enc.load_state_dict(ckpt["enc"])
        self.dec.load_state_dict(ckpt["dec"])

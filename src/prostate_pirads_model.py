from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_CONFIG = {
    "in_channels": 3,
    "latent_dim": 32,
    "input_size": 128,
    "num_classes": 5,
}

PIRADS_CLASS_NAMES = {
    0: "PI-RADS 1",
    1: "PI-RADS 2",
    2: "PI-RADS 3",
    3: "PI-RADS 4",
    4: "PI-RADS 5",
}


class PhysicsAwareSliceVAE(nn.Module):
    """Convolutional VAE with a deterministic multiclass PI-RADS head."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 32,
        input_size: int = 128,
        num_classes: int = 5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.feature_size = input_size // 16
        self.flat_dim = 128 * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(z.size(0), 128, self.feature_size, self.feature_size)
        return self.decoder(h)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        return self.classifier(mu)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(mu)
        return recon, mu, logvar, logits, z


def physics_aware_vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.01,
    alpha: float = 0.5,
    class_weights: torch.Tensor | None = None,
):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # cls_loss = F.cross_entropy(logits, target.long().view(-1), weight=class_weights)

    adc_mean = recon_x[:, 1].mean(dim=(1, 2))
    b1500_mean = recon_x[:, 2].mean(dim=(1, 2))
    # consistency_loss = torch.mean((adc_mean - b1500_mean) ** 2)

    total = recon_loss + beta * kl_loss + alpha * cls_loss + 0.1 * consistency_loss
    return total, {
        "recon_loss": float(recon_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "consistency_loss": float(consistency_loss.detach().cpu()),
    }


def build_checkpoint_payload(
    model: nn.Module,
    metrics: dict[str, Any] | None = None,
    class_weights: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "model_config": DEFAULT_MODEL_CONFIG.copy(),
        "class_names": PIRADS_CLASS_NAMES.copy(),
        "class_weights": class_weights,
        "metrics": metrics or {},
        "preprocessing": {
            "sequences": ["AX_T2", "AX_DIFFUSION_ADC", "AX_DIFFUSION_CALC_BVAL"],
            "input_size": 128,
            "center_crop_fraction": 0.88,
            "normalization": "per-slice percentile clip 1-99 then scale to 0-1",
        },
        "target": {
            "label": "joint_pirads",
            "class_index": "joint_pirads - 1",
            "patient_aggregation": "max slice expected PI-RADS score",
        },
    }

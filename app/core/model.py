from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_MODEL_CONFIG = {
    "in_channels": 3,
    "latent_dim": 32,
    "input_size": 128,
    "num_classes": 5,
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


def build_model_from_checkpoint(checkpoint: dict) -> PhysicsAwareSliceVAE:
    config = DEFAULT_MODEL_CONFIG.copy()
    config.update(checkpoint.get("model_config", {}))
    model = PhysicsAwareSliceVAE(**config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model

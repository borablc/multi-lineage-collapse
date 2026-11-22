"""Variational autoencoder for Fashion-MNIST.

This module: ELBO terms and reparameterization, the model, sampling and loading.
Reference: Kingma & Welling 2014.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SHAPE = (1, 28, 28)
HIDDEN_DIM = 400
LATENT_DIM = 20
CONV_HIDDEN_DIM = 256


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, exp(logvar)) || N(0, I) ) per sample, in nats.

    (B, D), (B, D) -> (B,)

    Closed form, summed over latent dims:
        0.5 * sum_d (mu^2 + exp(logvar) - 1 - logvar)
    """
    assert mu.shape == logvar.shape, f"mu {mu.shape} vs logvar {logvar.shape}"
    return ((mu ** 2 + torch.exp(logvar) - 1 - logvar) * 0.5).sum(dim=1)


def bernoulli_nll(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reconstruction term: -log p(x | z) under a per-pixel Bernoulli, in nats.

    (B, ...), (B, ...) -> (B,). logits and x must have the same shape;
    x is in [0, 1].
    """
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='none')
    return recon.flatten(start_dim=1).sum(dim=1)


def nelbo(logits: torch.Tensor, x: torch.Tensor,
          mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, torch.Tensor]:
    """Negative ELBO per sample, with both terms kept for logging.

    Returns {"recon": (B,), "kl": (B,), "nelbo": (B,)}.
    """
    results = {}
    results["recon"] = bernoulli_nll(logits, x)
    results["kl"] = gaussian_kl(mu, logvar)
    results["nelbo"] = results["recon"] + results["kl"]
    return results


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Draws z = mu + sigma * eps with eps ~ N(0, I), keeping gradients to mu and logvar.

    (B, D), (B, D) -> (B, D)

    sigma = exp(0.5 * logvar), because logvar = ln(sigma^2).
    Pass a seeded generator to make eps reproducible (needed for eval ELBO).
    """
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return mu + sigma * eps


class BaseVAE(nn.Module):
    """Shared VAE logic. Subclasses implement encode and decode."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 1, H, W) float in [0, 1] -> mu (B, D), logvar (B, D)."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D) -> logits (B, 1, H, W)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, 1, H, W) -> logits (B, 1, H, W), mu (B, D), logvar (B, D)."""
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar, generator=generator)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draws n images from the prior: z ~ N(0, I), then the decoder's mean image.

        Returns (n, 1, H, W) pixel intensities in [0, 1] (sigmoid of the logits;
        no Bernoulli coin flip). Pass a seeded generator on the model's device
        for reproducible samples.
        """
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, generator=generator, device=device)
        return self.decode(z).sigmoid()


class MLPVAE(BaseVAE):
    """MLP VAE with a Bernoulli decoder (outputs logits) and a diagonal Gaussian encoder."""

    def __init__(self, image_shape: tuple[int, ...] = IMAGE_SHAPE,
                 hidden_dim: int = HIDDEN_DIM, latent_dim: int = LATENT_DIM):
        super().__init__(latent_dim)

        self.image_shape = image_shape
        self.hidden_dim = hidden_dim
        image_dim = math.prod(image_shape)

        self.enc_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU()
        )

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, *image_shape) float in [0, 1] -> mu (B, D), logvar (B, D)."""
        h = self.enc_body(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D) -> logits (B, *image_shape)."""
        return self.decoder(z).view(-1, *self.image_shape)


class ConvVAE(BaseVAE):
    """Conv encoder (stride-2 convs, 28 -> 14 -> 7) and transposed-conv decoder (7 -> 14 -> 28).

    Only for 28x28 single-channel images: the 64 * 7 * 7 size is fixed by the strides.
    """

    def __init__(self, hidden_dim: int = CONV_HIDDEN_DIM, latent_dim: int = LATENT_DIM):
        super().__init__(latent_dim)
        self.hidden_dim = hidden_dim

        self.enc_body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),  # (B, 64, 7, 7)
            nn.Flatten(),  # (B, 3136)
            nn.Linear(64 * 7 * 7, hidden_dim), nn.ReLU(),  # (B, hidden)
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),  # (B, hidden)
            nn.Linear(hidden_dim, 64 * 7 * 7), nn.ReLU(),  # (B, 3136)
            nn.Unflatten(1, (64, 7, 7)),  # (B, 64, 7, 7)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (B, 1, 28, 28) logits
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 1, 28, 28) float in [0, 1] -> mu (B, D), logvar (B, D)."""
        h = self.enc_body(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D) -> logits (B, 1, 28, 28)."""
        return self.decoder(z)


ARCHITECTURES = {"mlp": MLPVAE, "conv": ConvVAE}


def build_vae(arch: str) -> BaseVAE:
    """Fresh, untrained VAE of the given architecture with default sizes."""
    if arch not in ARCHITECTURES:
        raise KeyError(f"unknown arch {arch!r}, expected one of {list(ARCHITECTURES)}")

    return ARCHITECTURES[arch]()


def load_vae(path, arch: str, device: str = "cuda") -> BaseVAE:
    """Loads a trained VAE in eval mode with all gradients disabled (frozen)."""
    model = build_vae(arch)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

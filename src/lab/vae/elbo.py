"""ELBO terms and the reparameterization trick for the VAEs.

NELBO = recon + KL, per sample, in nats:
  - recon: -log p(x | z) under a per-pixel Bernoulli (BCE with logits),
    summed over pixels
  - KL: closed-form KL( N(mu, exp(logvar)) || N(0, I) ), summed over latent dims
No batch reduction: callers take .mean() for training and keep per-sample
values for evaluation.

Reference: Kingma & Welling 2014.
"""
import torch
import torch.nn.functional as F


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

"""Unit tests for lab.vae on hand-computed toy cases."""
from math import log

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from lab.vae import gaussian_kl, bernoulli_nll, nelbo, reparameterize, VAE


def test_kl_prior():
    mu = torch.tensor([[0.]])
    logvar = torch.tensor([[0.]])

    out = gaussian_kl(mu, logvar)
    expected = 0.

    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected, abs=1e-4)


def test_kl_first_case():
    mu = torch.tensor([[-1.]])
    logvar = torch.tensor([[log(0.25)]])

    out = gaussian_kl(mu, logvar)
    expected = 0.5 * (1 + 0.25 - 1 - log(0.25))

    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected)


def test_kl_multi_dimension():
    mu = torch.tensor([[-1., 0.]])
    logvar = torch.tensor([[log(0.5 ** 2), log(1. ** 2)]])

    out = gaussian_kl(mu, logvar)
    expected = 0.5 * (1 + 0.25 - 1 - log(0.25))

    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected)


def test_kl_matches_torch_distributions():
    gen = torch.Generator().manual_seed(0)
    mu = torch.randn(2, 3, generator=gen)
    logvar = torch.randn(2, 3, generator=gen)

    sigma = torch.exp(0.5 * logvar)
    q = Normal(mu, sigma)
    prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    expected = kl_divergence(q, prior).sum(dim=-1)
    out = gaussian_kl(mu, logvar)

    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_recon_first_case():
    # Recon, case 1: x = [1, 0, 1], p = [0.99, 0.01, 0.99]
    #   pixel 1: x=1 -> -log(0.99)
    #   pixel 2: x=0 -> -log(1 - 0.01) = -log(0.99)
    #   pixel 3: x=1 -> -log(0.99)

    x = torch.tensor([[1., 0., 1.]])
    logits = torch.logit(torch.tensor([[.99, .01, .99]]))

    out = bernoulli_nll(logits, x)
    expected = -3 * log(0.99)

    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected, rel=1e-5)


def test_recon_second_case():
    # Recon, case 2: p = 0.5 everywhere (logits = 0)
    #   every pixel -> -log(0.5) = log(2), regardless of x

    x = torch.tensor([[1., 0., 1.]])
    logits_zeros = torch.tensor([[0., 0., 0.]])

    out = bernoulli_nll(logits_zeros, x)
    expected = 3 * log(2)

    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected, rel=1e-5)


def test_bernoulli_nll_sums_over_image_dims():
    x = torch.zeros(2, 1, 28, 28)
    logits = torch.zeros(2, 1, 28, 28)

    out = bernoulli_nll(logits, x)

    assert out.shape == (2,)
    assert out.tolist() == pytest.approx([784 * log(2)] * 2, rel=1e-6)


def test_elbo_batch():
    x = torch.tensor([[1., 0., 1.], [1., 0., 1.]])
    logits = torch.logit(torch.tensor([[.99, .01, .99], [.5, .5, .5]]))
    mu = torch.tensor([[-1.], [0.]])
    logvar = torch.tensor([[log(.5 ** 2)], [log(1. ** 2)]])

    out = nelbo(logits, x, mu, logvar)

    # Row 1: mu=-1, sigma=0.5, p=[0.99, 0.01, 0.99]
    # Row 2: mu=0,  sigma=1,   p=0.5 everywhere (logits = 0)
    expected_kl = torch.tensor([0.5 * (1 + 0.25 - 1 - log(0.25)), 0.0])  # 0.818147, 0
    expected_recon = torch.tensor([-3 * log(0.99), 3 * log(2)])  # 0.030151, 2.079442
    expected_nelbo = expected_kl + expected_recon  # 0.848298, 2.079442

    for key, expected in [("kl", expected_kl), ("recon", expected_recon), ("nelbo", expected_nelbo)]:
        assert out[key].shape == (2,), f"{key} shape {out[key].shape}"
        assert torch.allclose(out[key], expected, rtol=1e-5, atol=1e-6), \
            f"{key}: got {out[key].tolist()}, expected {expected.tolist()}"


def test_reparameterize_shape():
    mu = torch.rand(2, 3)
    logvar = torch.rand(2, 3)

    out = reparameterize(mu, logvar)
    assert out.shape == (2, 3)


def test_reparameterize_near_zero_sigma():
    mu = torch.rand(2, 3)
    logvar = torch.full((2, 3), -50.)

    out = reparameterize(mu, logvar)
    assert out.shape == (2, 3)
    assert torch.allclose(out, mu, rtol=1e-5, atol=1e-6)


def test_reparameterize_reproducibility():
    generator = torch.Generator().manual_seed(0)
    mu = torch.rand(2, 3)
    logvar = torch.rand(2, 3)

    out1 = reparameterize(mu, logvar, generator=generator)

    generator = torch.Generator().manual_seed(0)
    out2 = reparameterize(mu, logvar, generator=generator)

    assert torch.equal(out1, out2)


def test_reparameterize_gradient_flows_to_mu():
    mu = torch.rand(2, 3, requires_grad=True)
    logvar = torch.rand(2, 3, requires_grad=True)

    z = reparameterize(mu, logvar)
    z.sum().backward()
    expected = torch.ones_like(mu)

    assert torch.equal(expected, mu.grad)


def test_vae_shapes_toys():
    model = VAE(image_shape=(1, 2, 3), hidden_dim=4, latent_dim=2)
    x = torch.rand(5, 1, 2, 3)
    logits, mu, logvar = model(x)

    assert logits.shape == (5, 1, 2, 3), f"{logits.shape} is not (5,1,2,3)"
    assert mu.shape == (5, 2), f"{mu.shape} is not (5,2)"
    assert logvar.shape == (5, 2), f"{logvar.shape} is not (5,2)"


def test_vae_shapes_actual():
    model = VAE()
    x = torch.rand(3, 1, 28, 28)
    logits, mu, logvar = model(x)
    nelbo_val = nelbo(logits, x, mu, logvar)["nelbo"]

    assert logits.shape == (3, 1, 28, 28), f"{logits.shape} is not (3,1,28,28)"
    assert mu.shape == (3, 20), f"{mu.shape} is not (3,20)"
    assert logvar.shape == (3, 20), f"{logvar.shape} is not (3,20)"
    assert nelbo_val.shape == (3,), f"{nelbo_val.shape} is not (3,)"


def test_param_count():
    model = VAE()

    expected = (784 * 400 + 400) + (400 * 20 + 20) * 2 + (20 * 400 + 400) + (400 * 784 + 784)
    out = sum(p.numel() for p in model.parameters())

    assert out == expected, f"{out} != {expected}"


def test_gradient_flows_to_model():
    model = VAE()
    x = torch.rand(3, 1, 28, 28)
    logits, mu, logvar = model(x)

    nelbo_val = nelbo(logits, x, mu, logvar)["nelbo"]
    nelbo_val.mean().backward()

    assert all(p.grad is not None for p in model.parameters())


def test_same_seed_gives_same_model_and_output():
    x = torch.rand(3, 1, 28, 28)

    torch.manual_seed(0)
    model_a = VAE()
    torch.manual_seed(0)
    model_b = VAE()

    assert all(torch.equal(p, q) for p, q in zip(model_a.parameters(), model_b.parameters()))

    generator = torch.Generator().manual_seed(1)
    logits_a, mu_a, logvar_a = model_a(x, generator=generator)

    generator = torch.Generator().manual_seed(1)
    logits_b, mu_b, logvar_b = model_b(x, generator=generator)

    assert torch.equal(logits_a, logits_b), f"{logits_a} != {logits_b}"


def test_generator_controls_sampling():
    x = torch.rand(3, 1, 28, 28)
    model = VAE()

    logits_1, _, _ = model(x, generator=torch.Generator().manual_seed(1))
    logits_2, _, _ = model(x, generator=torch.Generator().manual_seed(2))

    assert not torch.equal(logits_1, logits_2)


def test_sample_size():
    model = VAE(image_shape=(1, 2, 3), hidden_dim=4, latent_dim=2)

    out = model.sample(7)
    expected = (7, 1, 2, 3)

    assert out.shape == expected, f"out shape {out.shape} != {expected}"


def test_sample_values():
    model = VAE(image_shape=(1, 2, 3), hidden_dim=4, latent_dim=2)

    out = model.sample(10)

    assert ((out >= 0) & (out <= 1)).all()


def test_sample_repeatability():
    model = VAE(image_shape=(1, 2, 3), hidden_dim=4, latent_dim=2)

    generator = torch.Generator().manual_seed(6)
    out_1 = model.sample(10, generator=generator)

    generator = torch.Generator().manual_seed(6)
    out_2 = model.sample(10, generator=generator)

    assert torch.equal(out_1, out_2), f"{out_1} != {out_2}"


def test_sample_requires_no_grad():
    model = VAE(image_shape=(1, 2, 3), hidden_dim=4, latent_dim=2)
    out = model.sample(1)

    assert not out.requires_grad, f"{out.requires_grad}"

"""Single VAE training run on real Fashion-MNIST"""
import json
from datetime import datetime
import time

import torch

from lab.data import load_fmnist, preprocess, quantize
from lab.utils import set_seed, git_info
from lab.vae.model import load_vae, build_vae, BaseVAE
from lab.vae.elbo import nelbo
from lab.paths import CHECKPOINT_DIR

SEED = 0
EPOCHS = 50  # frozen: chosen from 200-epoch constant-lr curves on Gen0 real data (mlp and conv)
BATCH_SIZE = 128
LR = 1e-3
EVAL_SEED = 2024
KEYS = ("recon", "kl", "nelbo")
GEN_CHUNK = 10_000
SCHEDULE = "cosine"  # "cosine" or "constant"
SAVE_CHECKPOINT = True  # False for measurement runs, so the real checkpoint is not overwritten

ARCH = "conv"  # "mlp" or "conv", see lab.vae.ARCHITECTURES


def train_one_epoch(model: BaseVAE, optimizer: torch.optim.Optimizer,
                    x_train: torch.Tensor, batch_size: int) -> dict[str, float]:
    """One shuffled pass over x_train ((N, 28, 28) uint8, on the model's device).

    Forward in bf16 autocast, loss in fp32. Returns mean recon, kl, nelbo per image.
    """
    model.train()
    perm = torch.randperm(len(x_train), device=x_train.device)
    totals = {k: torch.zeros((), device=x_train.device) for k in KEYS}

    for i in range(0, len(x_train), batch_size):
        xb = preprocess(x_train[perm[i:i + batch_size]])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, mu, logvar = model(xb)
        out = nelbo(logits.float(), xb, mu.float(), logvar.float())
        loss = out["nelbo"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k in KEYS:
            totals[k] += out[k].detach().sum()

    return {k: (totals[k] / len(x_train)).item() for k in KEYS}  # (one sync per epoch)


@torch.no_grad()
def evaluate_nelbo(model: BaseVAE, x: torch.Tensor, batch_size: int = 1000,
                   seed: int = EVAL_SEED) -> dict[str, float]:
    """Mean recon, kl, nelbo per image on x ((N, 28, 28) uint8), fp32, fixed eps."""
    model.eval()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    totals = {k: torch.zeros((), device=x.device) for k in KEYS}
    for i in range(0, len(x), batch_size):
        xb = preprocess(x[i:i + batch_size])
        logits, mu, logvar = model(xb, generator=generator)
        out = nelbo(logits, xb, mu, logvar)
        for k in KEYS:
            totals[k] += out[k].detach().sum()

    return {k: (totals[k] / len(x)).item() for k in KEYS}


@torch.no_grad()
def generate_uint8(model: BaseVAE, n: int, generator: torch.Generator,
                   chunk: int = GEN_CHUNK) -> torch.Tensor:
    """n prior samples as (n, 28, 28) uint8, the form they enter the pool.

    Generated in chunks so 60k samples do not need one big tensor on the GPU.
    All latents are drawn at once, so the chunk size does not change which images
    are generated. Matmul summation order can depend on the batch size, so a few
    pixels may differ by one gray level; keep chunk fixed for bit-exact runs.
    """
    device = next(model.parameters()).device
    z = torch.randn(n, model.latent_dim, generator=generator, device=device)
    out = []
    for i in range(0, len(z), chunk):
        out.append(quantize(model.mean_image(z[i:i + chunk])))
    return torch.cat(out)


def train_vae(x_uint8: torch.Tensor, arch: str, seed: int, epochs: int = EPOCHS) -> BaseVAE:
    """Fresh VAE trained with the official recipe (Adam, cosine, bf16 forward) on x_uint8.

    x_uint8: (N, 28, 28) on the training device. Quiet (no per-epoch printing).
    Returns the model in eval mode.
    """
    set_seed(seed)
    model = build_vae(arch).to(x_uint8.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        train_one_epoch(model, optimizer, x_uint8, BATCH_SIZE)
        scheduler.step()

    model.eval()
    return model


def main():
    info = git_info()
    if info["dirty"]:
        print("WARNING: the code has uncommitted changes")

    device = "cuda"
    set_seed(SEED)
    data = load_fmnist(device)
    model = build_vae(ARCH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if SCHEDULE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("epoch|train_nelbo|test_recon| test_kl  |test_nelbo| time")
    train_zero = evaluate_nelbo(model, data["x_train"])
    test_zero = evaluate_nelbo(model, data["x_test"])
    print(
        f"   0 | {train_zero["nelbo"]:9.2f} | {test_zero["recon"]:8.2f} | {test_zero["kl"]:8.2f} | {test_zero["nelbo"]:8.2f} | 0.00")

    for epoch in range(1, EPOCHS + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        train = train_one_epoch(model, optimizer, data["x_train"], BATCH_SIZE)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        test = evaluate_nelbo(model, data["x_test"])
        print(
            f"{epoch:4d} | {train["nelbo"]:9.2f} | {test["recon"]:8.2f} | {test["kl"]:8.2f} | {test["nelbo"]:8.2f} | {dt:.2f}")
        if SCHEDULE == "cosine":
            scheduler.step()

    if SAVE_CHECKPOINT:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"vae_fmnist_{ARCH}_seed{SEED}.pt")
        meta = {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "schedule": SCHEDULE,
            "arch": ARCH,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
            "final_test": test,  # (the last epoch's recon / kl / nelbo dict)
            "git": info,
            "torch": torch.__version__,
            "created_at": datetime.now().isoformat(timespec="seconds")
        }

        with open(CHECKPOINT_DIR / f"vae_fmnist_{ARCH}_seed{SEED}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # reload sanity check:
        model = load_vae(CHECKPOINT_DIR / f"vae_fmnist_{ARCH}_seed{SEED}.pt", ARCH)
        reload_test = evaluate_nelbo(model, data["x_test"])
        assert test["nelbo"] == reload_test["nelbo"], f"{test['nelbo']} != {reload_test['nelbo']}"
        print(f"[PASSED] reloaded model gives identical test NELBO ({reload_test['nelbo']:.4f})")


if __name__ == "__main__":
    main()

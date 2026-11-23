"""Exploration: N = 1 demo pilot, replace vs accumulate, MLP vs conv VAE.

Not the planned pilot: one seed, 5 generations, first look only. Kept in git
because its numbers are shown to the advisor. Outputs go to explorations/demo_pilot/.
"""
import csv
import time

import torch
from torchvision.utils import save_image

from lab.data import load_fmnist, preprocess
from lab.judge.model import extract_features, load_judge, JudgeCNN
from lab.metrics.core import evaluate
from lab.paths import CHECKPOINT_DIR, PROJECT_ROOT
from lab.vae.model import BaseVAE, load_vae
from lab.vae.train import evaluate_nelbo, generate_uint8, train_vae

REGIME_IDS = {"replace": 1, "accumulate": 2}
CSV_COLUMNS = ("gen", "syn_frac", "coverage", "precision", "density", "recall",
               "fid", "class_entropy", "test_nelbo", "seconds")

OUT_DIR = PROJECT_ROOT / "explorations" / "demo_pilot"
GRID_DIR = OUT_DIR / "grids"
GEN_DIR = OUT_DIR / "generations"
METRIC_DIR = OUT_DIR / "metrics"

ARCHS = ("mlp", "conv")
REGIMES = ("accumulate", "replace")
N_GENS = 5
R = 60_000
M = 30_000  # M/R = 50%
N_EVAL = 3000  # same size as reference A
SAMPLE_SEED = 3110  # same as inspect_gen0, so Gen0 rows must match exactly
GRID_SEED = 541
GRID_SIZE = 64
ROW_SIZE = 8  # images per generation in the generations picture
DEVICE = "cuda"


def make_dirs() -> None:
    for d in (GRID_DIR, GEN_DIR, METRIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_real_grid(x_test: torch.Tensor) -> None:
    """The first GRID_SIZE test images, for side-by-side comparison."""
    save_image(preprocess(x_test[:GRID_SIZE]), GRID_DIR / "real.png", nrow=8)


def save_gen0_grid(model: BaseVAE, arch: str) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(GRID_SEED)
    model_grids = model.sample(GRID_SIZE, generator=generator)
    save_image(model_grids, GRID_DIR / f"{arch}_gen0.png", nrow=8)


def measure(model: BaseVAE, judge: JudgeCNN, ref: dict, x_test: torch.Tensor) -> dict[str, float]:
    """N_EVAL prior samples against reference A, plus real test NELBO."""
    generator = torch.Generator(device=DEVICE).manual_seed(SAMPLE_SEED)
    x = generate_uint8(model, N_EVAL, generator=generator)
    feat = extract_features(judge, x)
    labels = judge.labels(feat)
    row = evaluate(ref["feat_a"], feat, labels)
    row["test_nelbo"] = evaluate_nelbo(model, x_test)["nelbo"]
    return row


def next_training_set(regime: str, prev: BaseVAE, pool: torch.Tensor,
                      generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Training data for the next generation.

    replace:    R fresh samples from prev, nothing else (pool unused)
    accumulate: add M samples from prev to the pool, then draw R uniformly from it
    Returns (x_train, new pool, synthetic fraction of x_train).
    Real images are the first R rows of the pool.
    """
    if regime == "replace":
        x = generate_uint8(prev, R, generator=generator)
        return x, pool, 1.0
    elif regime == "accumulate":
        pool = torch.cat([pool, generate_uint8(prev, M, generator=generator)])
        idx = torch.randperm(len(pool), generator=generator, device=pool.device)[:R]
        return pool[idx], pool, (idx >= R).float().mean().item()
    else:
        raise ValueError(f"Unknown regime: {regime}")


def generation_row(model: BaseVAE) -> torch.Tensor:
    """ROW_SIZE samples for the per-generation picture."""
    generator = torch.Generator(device=DEVICE).manual_seed(GRID_SEED)
    return model.sample(ROW_SIZE, generator=generator)


def write_csv(rows: list[dict], path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_regime(arch: str, regime: str, gen0: BaseVAE, gen0_row: dict,
               data: dict, judge: JudgeCNN, ref: dict) -> None:
    rid = REGIME_IDS[regime]
    rows = [{"gen": 0, "syn_frac": 0.0, **gen0_row, "seconds": 0.0}]
    pictures = [generation_row(gen0)]
    pool = data["x_train"].clone()
    prev = gen0
    print(f"\n{arch} {regime}: gen 0 coverage {gen0_row['coverage']:.4f}")

    for t in range(1, N_GENS + 1):
        t0 = time.perf_counter()
        generator = torch.Generator(device=DEVICE).manual_seed(10_000 * rid + t)
        x_train, pool, syn = next_training_set(regime, prev, pool, generator)
        model = train_vae(x_train, arch, seed=100 * rid + t)
        row = {"gen": t, "syn_frac": syn, **measure(model, judge, ref, data["x_test"]),
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        pictures.append(generation_row(model))
        prev = model
        print(
            f"  gen {t}  syn {syn:.3f}  coverage {row['coverage']:.4f} nelbo {row['test_nelbo']:.2f}  {row['seconds']:.0f}s")

    write_csv(rows, METRIC_DIR / f"{arch}_{regime}.csv")
    save_image(torch.cat(pictures), GEN_DIR / f"{arch}_{regime}.png", nrow=ROW_SIZE)


def main():
    make_dirs()
    data = load_fmnist(DEVICE)
    judge = load_judge(CHECKPOINT_DIR / "judge_fmnist.pt", DEVICE)
    ref = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")

    save_real_grid(data["x_test"])
    for arch in ARCHS:
        gen0 = load_vae(CHECKPOINT_DIR / f"vae_fmnist_{arch}_seed0.pt", arch, DEVICE)
        save_gen0_grid(gen0, arch)
        gen0_row = measure(gen0, judge, ref, data["x_test"])
        for regime in REGIMES:
            run_regime(arch, regime, gen0, gen0_row, data, judge, ref)


if __name__ == "__main__":
    main()

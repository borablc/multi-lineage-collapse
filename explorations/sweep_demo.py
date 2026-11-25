"""Demo sweep for the advisor: conv VAE, accumulate pool, N in {1, 2}, M/R in {10%, 50%}, 3 seeds.

Purpose: show the FORMAT of the thesis tables. 3 seeds and 3 generations are far too few
for conclusions; numbers are illustrative only.
"""
import csv
import time
import statistics

import torch
import matplotlib.pyplot as plt

from lab.data import load_fmnist
from lab.judge.model import extract_features, load_judge
from lab.metrics.core import evaluate
from lab.paths import CHECKPOINT_DIR, PROJECT_ROOT
from lab.vae.model import BaseVAE
from lab.vae.train import evaluate_nelbo, generate_uint8, train_vae

ARCH = "conv"
NS = (1, 2)
MR_RATIOS = (0.10, 0.50)
SEEDS = (0, 1, 2)
N_GENS = 3  # generations after Gen0, so Gen0...Gen3 are trained and measured
R = 60_000
N_EVAL = 3000  # ecosystem output measured per generation, split evenly across the N models
DEVICE = "cuda"
EPOCHS = 50  # 2 for debugging slices, 10 for the dry run, 50 (frozen recipe) for the real demo
OUT_DIR = PROJECT_ROOT / "explorations" / f"sweep_demo_ep{EPOCHS}"
PURPOSES = {"train": 0, "draw": 1, "generate": 2, "measure": 3}
CSV_COLUMNS = ("gen", "syn_frac", "coverage", "precision", "density", "recall",
               "fid", "class_entropy", "test_nelbo", "seconds")
COVERAGE_REAL = 0.9747  # A vs B, sanity baseline (step 2)
COVERAGE_NOISE = 0.0137  # A vs uniform noise, sanity scenario (step 2)
LINE_COLORS = {1: "tab:blue", 2: "tab:orange"}  # N -> color
LINE_STYLES = {1: "-", 2: "--"}  # mr_id -> line style


def make_seed(seed: int, n: int, mr_id: int, gen: int, k: int, purpose: str) -> int:
    """Deterministic, unique integer seed for one (seed, N, M/R, gen, model, purpose) step.

    mr_id is the index into MR_RATIOS plus 1; Gen0 uses mr_id = 0 because it is shared by all M/R levels.
    purpose separates the random streams of one step (train, draw, generate, measure),
    so e.g. training and sampling of the same model never share a seed.
    Each field gets its own decimal digits so no two steps collide:
        seed * 1_000_000 + n * 100_000 + mr_id * 10_000 + gen * 100 + k * 10 + PURPOSES[purpose]
    """
    assert n < 10 and mr_id < 10 and k < 10 and gen < 100
    return seed * 1_000_000 + n * 100_000 + mr_id * 10_000 + gen * 100 + k * 10 + PURPOSES[purpose]


def measure(models: list[BaseVAE], judge, feat_a: torch.Tensor,
            x_test: torch.Tensor, generator: torch.Generator) -> dict[str, float]:
    """Measures the ecosystem output of one generation.

    Each model generates N_EVAL // len(models) images (uint8), they are concatenated
    into one (N_EVAL, 28, 28) set and evaluated against reference A.
    test_nelbo is the mean of evaluate_nelbo(model, x_test)["nelbo"] over the models.
    """
    assert N_EVAL % len(models) == 0
    out = []
    total_nelbo = 0.0
    for model in models:
        out.append(generate_uint8(model, N_EVAL // len(models), generator))
        total_nelbo += evaluate_nelbo(model, x_test)["nelbo"]
    samples = torch.cat(out)
    feats = extract_features(judge, samples)
    labels = judge.labels(feats)
    evals = evaluate(feat_a, feats, labels)
    evals["test_nelbo"] = total_nelbo / len(models)

    return evals


def new_pool(x_real: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pool with only real data. All three tensors live on x_real's device and share length P.

    x: (P, 28, 28) uint8; source: (P,) int64, -1 = real, k = made by model k; gen: (P,) int64, -1 for real.
    """
    tags = torch.full((len(x_real),), -1, dtype=torch.int64, device=x_real.device)
    return {"x": x_real, "source": tags, "gen": tags.clone()}


def add_to_pool(pool: dict[str, torch.Tensor], x_new: torch.Tensor, source: int, gen: int) -> None:
    """Appends x_new (uint8, same device) to the pool in place, tagged with its source model and generation."""
    assert x_new.dtype == torch.uint8
    n_new = len(x_new)
    device = pool["x"].device
    pool["x"] = torch.cat((pool["x"], x_new))
    pool["source"] = torch.cat((pool["source"], torch.full((n_new,), source, dtype=torch.int64, device=device)))
    pool["gen"] = torch.cat((pool["gen"], torch.full((n_new,), gen, dtype=torch.int64, device=device)))


def next_generation(pool: dict[str, torch.Tensor], prev_models: list[BaseVAE], x_seed: int,
                    n: int, mr_id: int, gen: int) -> tuple[list[BaseVAE], float]:
    """One accumulate step: Gen (gen - 1) models write to the pool, then n fresh Gen gen models read from it.

    x_seed is the run seed (0, 1, 2). Returns the new models and the mean synthetic fraction
    of their training draws.
    """
    assert mr_id >= 1, "mr_id 0 is reserved for Gen0; MR_RATIOS[-1] would silently pick the last ratio"
    assert len(prev_models) == n, f"expected {n} previous models, got {len(prev_models)}"
    m = round(MR_RATIOS[mr_id - 1] * R)
    assert m % n == 0, f"M = {m} must split evenly across {n} models"

    # Populate the pool with previous models
    for k, prev in enumerate(prev_models):
        g_seed = make_seed(x_seed, n, mr_id, gen, k, "generate")
        g = torch.Generator(device=DEVICE).manual_seed(g_seed)
        x_new = generate_uint8(prev, m // n, g)
        add_to_pool(pool, x_new, k, gen - 1)

    # Train new models
    total_syn_frac = 0.0
    new_models = []
    for k in range(n):
        g_seed = make_seed(x_seed, n, mr_id, gen, k, "draw")
        g = torch.Generator(device=DEVICE).manual_seed(g_seed)
        idx = torch.randperm(len(pool["x"]), generator=g, device=DEVICE)[:R]
        total_syn_frac += (pool["source"][idx] >= 0).float().mean().item()

        t_seed = make_seed(x_seed, n, mr_id, gen, k, "train")
        new_models.append(train_vae(pool["x"][idx], ARCH, t_seed, epochs=EPOCHS))

    return new_models, total_syn_frac / n


def train_gen0(x_train: torch.Tensor, seed: int, n: int) -> list[BaseVAE]:
    """Trains the N Gen0 models on real data only. Shared by every M/R level of (seed, n)."""
    vaes = []
    for k in range(n):
        seed_k = make_seed(seed, n, 0, 0, k, "train")
        vaes.append(train_vae(x_train, ARCH, seed_k, epochs=EPOCHS))
    return vaes


def csv_path(seed: int, n: int, mr_id: int):
    """OUT_DIR / "runs" / "n2_mr50_seed0.csv" style path for one configuration."""
    pct = round(MR_RATIOS[mr_id - 1] * 100)
    return OUT_DIR / "runs" / f"n{n}_mr{pct}_seed{seed}.csv"


def write_csv(path, rows: list[dict]) -> None:
    """Writes rows with CSV_COLUMNS as header. LF line endings (Python's csv default is CRLF)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def timer() -> float:
    """Wall-clock time after waiting for queued GPU work, so measured durations are real."""
    torch.cuda.synchronize()
    return time.perf_counter()


def run_config(data, judge, feat_a, seed: int, n: int, mr_id: int,
               gen0_models: list[BaseVAE], gen0_row: dict) -> list[dict]:
    """Runs Gen1..Gen N_GENS of one (seed, n, M/R) configuration on a fresh pool. Returns all rows incl. Gen0."""
    pool = new_pool(data["x_train"])
    models = gen0_models
    rows = [gen0_row]
    pct = round(MR_RATIOS[mr_id - 1] * 100)

    for gen in range(1, N_GENS + 1):
        start = timer()

        models, syn = next_generation(pool, models, seed, n, mr_id, gen)

        g_seed = make_seed(seed, n, mr_id, gen, 0, "measure")
        g = torch.Generator(device=DEVICE).manual_seed(g_seed)
        row = measure(models, judge, feat_a, data["x_test"], g)
        row.update(gen=gen, syn_frac=syn, seconds=timer() - start)
        print(
            f"seed: {seed}, n: {n}, mr: {pct}, gen: {row['gen']}, coverage: {row['coverage']:.4f}, "
            f"syn_frac: {row['syn_frac']:.4f}, seconds: {row['seconds']:.2f}")
        rows.append(row)
    return rows


def load_csv(path) -> list[dict[str, float]]:
    """Reads one configuration CSV back as a list of rows with float values, ordered by gen."""
    with open(path, newline="") as f:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    return sorted(rows, key=lambda row: row["gen"])


def cell(values: list[float], decimals: int = 1, signed: bool = True) -> str:
    """Formats per-seed values as 'mean ± std' (sample std, n - 1); signed adds an explicit +/- for changes."""
    sign = "+" if signed else ""
    return f"{statistics.mean(values):{sign}.{decimals}f} ± {statistics.stdev(values):.{decimals}f}"


def draw_curves(ax, runs: dict, pcts: list[int]) -> None:
    """Mean Coverage per generation for every (N, M/R), with a shaded mean ± std band over seeds.

    runs maps (n, mr_id) -> list of per-seed row lists; pcts are the M/R levels in percent.
    """
    gens = list(range(N_GENS + 1))
    for (n, mr_id), seed_rows in runs.items():
        cov = [[r["coverage"] for r in rows] for rows in seed_rows]  # (seeds, gens)
        per_gen = list(zip(*cov))  # (gens, seeds)
        mean = [statistics.mean(v) for v in per_gen]
        std = [statistics.stdev(v) for v in per_gen]
        ax.plot(gens, mean, color=LINE_COLORS[n], linestyle=LINE_STYLES[mr_id], marker="o",
                label=f"N={n}, M/R={pcts[mr_id - 1]}%")
        ax.fill_between(gens, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                        color=LINE_COLORS[n], alpha=0.15)
    ax.set_xticks(gens)
    ax.set_xlabel("generation")
    ax.set_ylabel("Coverage (k = 5, vs reference A)")


def summarize() -> None:
    """Builds the advisor tables (markdown) and the Coverage plots from all CSVs in OUT_DIR / "runs"."""
    runs = {}  # (n, mr_id) -> list of per-seed row lists
    for n in NS:
        for mr_id in range(1, len(MR_RATIOS) + 1):
            runs[(n, mr_id)] = [load_csv(csv_path(s, n, mr_id)) for s in SEEDS]
    pcts = [round(r * 100) for r in MR_RATIOS]

    # Tables: title -> (per-seed value function, decimals, signed)
    baselines = f"real A vs B: {COVERAGE_REAL}, noise: {COVERAGE_NOISE}"
    tables = {
        f"Coverage at Gen0 ({baselines})":
            (lambda rows: rows[0]["coverage"], 3, False),
        "Relative Coverage change, Gen0 -> Gen3 (%)":
            (lambda rows: 100 * (rows[-1]["coverage"] - rows[0]["coverage"]) / rows[0]["coverage"], 1, True),
        f"Coverage at Gen3 ({baselines})":
            (lambda rows: rows[-1]["coverage"], 3, False),
        "Test NELBO change, Gen3 - Gen0 (nats)":
            (lambda rows: rows[-1]["test_nelbo"] - rows[0]["test_nelbo"], 1, True),
    }

    lines = [f"Demo, {len(SEEDS)} seeds, {EPOCHS} epochs, mean ± std (sample std). Format only, not results.", ""]
    for title, (fn, dec, signed) in tables.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| N \\ M/R | " + " | ".join(f"{p}%" for p in pcts) + " |")
        lines.append("|---" * (len(pcts) + 1) + "|")
        for n in NS:
            cells = [cell([fn(rows) for rows in runs[(n, mr_id)]], dec, signed)
                     for mr_id in range(1, len(MR_RATIOS) + 1)]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")

    text = "\n".join(lines)
    (OUT_DIR / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)

    # Plots: full scale with baselines (honest position) next to a zoom (dynamics across generations)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))

    draw_curves(ax_full, runs, pcts)
    ax_full.axhline(COVERAGE_REAL, color="tab:green", linestyle=":", label=f"real A vs B ({COVERAGE_REAL})")
    ax_full.axhline(COVERAGE_NOISE, color="tab:red", linestyle=":", label=f"noise ({COVERAGE_NOISE})")
    ax_full.set_ylim(0, 1)
    ax_full.set_title("Full scale, with real and noise baselines")
    ax_full.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.94))

    draw_curves(ax_zoom, runs, pcts)
    ax_zoom.set_title("Zoomed (y axis does not start at 0)")
    ax_zoom.legend(fontsize=8)

    fig.suptitle(f"Demo: conv VAE, accumulate, {len(SEEDS)} seeds, {EPOCHS} epochs")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_curves.png", dpi=150)
    plt.close(fig)


def main():
    data = load_fmnist(device=DEVICE)
    judge = load_judge(CHECKPOINT_DIR / "judge_fmnist.pt", device=DEVICE)
    feat_a = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")["feat_a"]

    for seed in SEEDS:
        for n in NS:
            paths = {mr_id: csv_path(seed, n, mr_id) for mr_id in range(1, len(MR_RATIOS) + 1)}
            if all(p.exists() for p in paths.values()):
                print(f"[SKIPPED] seed {seed}, n {n}: all {len(paths)} CSVs exist")
                continue
            start = timer()
            gen0_models = train_gen0(data["x_train"], seed, n)
            g_seed = make_seed(seed, n, 0, 0, 0, "measure")
            g = torch.Generator(device=DEVICE).manual_seed(g_seed)
            gen0_row = measure(gen0_models, judge, feat_a, data["x_test"], g)
            gen0_row.update(gen=0, syn_frac=0.0, seconds=timer() - start)

            for mr_id in range(1, len(MR_RATIOS) + 1):
                if paths[mr_id].exists():
                    continue

                pct = round(MR_RATIOS[mr_id - 1] * 100)
                print(f"seed: {seed}, n: {n}, mr: {pct}, gen: 0, coverage: {gen0_row['coverage']:.4f}, "
                      f"syn_frac: {gen0_row['syn_frac']:.4f}, seconds: {gen0_row['seconds']:.2f}")

                rows = run_config(data, judge, feat_a, seed, n, mr_id, gen0_models, gen0_row)
                write_csv(paths[mr_id], rows)
    summarize()


if __name__ == "__main__":
    main()

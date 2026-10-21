"""Sanity checks for the metric module on fake sets with known answers.

Scenarios (real reference is always set A, 3000 samples):
  1. B:     a second real set from the same distribution (perfect generator)
  2. noise: uniform random pixels (generator that produces garbage)
  3. B with one class removed (generator that forgot a mode)
"""
import math

import torch

from lab.judge import NUM_CLASSES, extract_features, load_judge
from lab.metrics import (DEFAULT_K, _inside, class_entropy, fid, knn_radii,
                         pairwise_distances, prdc)
from lab.paths import CHECKPOINT_DIR
from lab.data import load_fmnist

NOISE_SEED = 9001  # IT'S OVER 9000!
DROP_SEED = 1984

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
DROPPED_CLASS = 8

TABLE_COLUMNS = ["coverage", "recall", "precision", "density", "fid", "class_entropy"]


def check(condition: bool, description: str) -> None:
    """Assert a sanity condition and report it by name."""
    assert condition, f"FAILED: {description}"
    print(f"[PASSED] {description}")


def evaluate(real_feat: torch.Tensor, fake_feat: torch.Tensor,
             fake_labels: torch.Tensor) -> dict[str, float]:
    """All metrics of one fake set against the real reference."""
    metrics_dic = prdc(real_feat, fake_feat)
    metrics_dic["fid"] = fid(real_feat, fake_feat)
    metrics_dic["class_entropy"] = class_entropy(fake_labels, NUM_CLASSES)
    return metrics_dic


def noise_images(n: int, seed: int) -> torch.Tensor:
    """(n, 28, 28) uint8, every pixel uniform in 0..255."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 28, 28), dtype=torch.uint8, generator=g)


def drop_class_set(feat_b: torch.Tensor, idx_b: torch.Tensor, idx_a: torch.Tensor,
                   x_test: torch.Tensor, y_test: torch.Tensor,
                   dropped: int, judge, seed: int, num_classes: int) -> torch.Tensor:
    """B with every sample of `dropped` removed, refilled to the same size
    from unused test images of the other classes. Returns (len(idx_b), D) features."""
    # Drop the class
    keep = y_test[idx_b] != dropped
    kept_feat = feat_b[keep]
    n_missing = len(idx_b) - len(kept_feat)
    assert n_missing > 0, f"class {dropped} not found in B"

    # Find unused test indices (not in idx_a, not in idx_b, not in dropped class)
    used = torch.zeros(len(y_test), dtype=torch.bool)
    used[idx_a] = True
    used[idx_b] = True
    candidates = ((~used) & (y_test != dropped)).nonzero().squeeze(1)

    # Pick n_missing (stratified)
    g = torch.Generator().manual_seed(seed)
    remaining = [c for c in range(num_classes) if c != dropped]
    base, extra = divmod(n_missing, len(remaining))
    refill = []
    for i, c in enumerate(remaining):
        n_c = base + 1 if i < extra else base
        class_candidates = candidates[y_test[candidates] == c]
        perm = torch.randperm(len(class_candidates), generator=g)
        refill.append(class_candidates[perm[:n_c]])
    refill_idx = torch.cat(refill)
    assert len(refill_idx) == n_missing, "not enough unused candidates to refill"

    refill_feat = extract_features(judge, x_test[refill_idx])
    return torch.cat([kept_feat, refill_feat])


def drop_class_sweep(feat_a: torch.Tensor, feat_b: torch.Tensor,
                     idx_a: torch.Tensor, idx_b: torch.Tensor,
                     x_test: torch.Tensor, y_test: torch.Tensor,
                     judge, baseline: dict[str, float]) -> dict[int, dict[str, float]]:
    """Drop each class in turn and print one row per class (report only, no checks).

    The first row is the A-B baseline for comparison. Returns {class: metrics}.
    """
    print("Drop-class sweep")
    header = format_header()
    print(header)
    print("-" * len(header))
    print(format_row("none (A-B)", baseline))

    sweep = {}
    for c in range(NUM_CLASSES):
        feat_drop = drop_class_set(feat_b, idx_b, idx_a, x_test, y_test,
                                   c, judge, DROP_SEED, NUM_CLASSES)
        labels_drop = judge.head(feat_drop).argmax(dim=1)
        sweep[c] = evaluate(feat_a, feat_drop, labels_drop)
        print(format_row(f"{c} {CLASS_NAMES[c]}", sweep[c]))
    print()
    return sweep


def format_row(label: str, metrics: dict[str, float], decimals: int = 6) -> str:
    """One table row: a left-aligned label, then one right-aligned column per metric."""
    return f"{label:<12}" + "".join(f"{metrics[col]:>15.{decimals}f}" for col in TABLE_COLUMNS)


def format_header() -> str:
    return f"{'scenario':<12}" + "".join(f"{col:>15}" for col in TABLE_COLUMNS)


def print_table(results: dict[str, dict[str, float]], decimals: int = 6) -> None:
    """One row per scenario, one column per metric."""
    header = format_header()
    print(header)
    print("-" * len(header))
    for scenario, metrics in results.items():
        print(format_row(scenario, metrics, decimals))
    print()


def diagnose_noise_precision(real_feat: torch.Tensor, real_labels: torch.Tensor,
                             noise_feat: torch.Tensor, noise_labels: torch.Tensor,
                             k: int = DEFAULT_K) -> None:
    """Explain why precision stays high on noise while density collapses.

    Not called by main. Kept to rerun the same diagnosis on standardized features.
    Findings on raw features (see notebook): 41 reals hold half of the noise
    inside their balls; noise radii are very uniform (one homogeneous cloud);
    the judge splits that cloud into shirt/bag with equal precision in both halves.
    """
    real_radii = knn_radii(real_feat, k)
    noise_radii = knn_radii(noise_feat, k)
    inside = _inside(real_radii, pairwise_distances(real_feat, noise_feat))  # (N_real, N_noise)

    print("--- Diagnostics: noise precision ---")

    # Which real balls do the work?
    swallowed = inside.sum(dim=1)  # per real: how many noise samples inside its ball
    culprits = (swallowed > 0).nonzero().squeeze(1)
    print("Reals whose ball contains any noise:", len(culprits))
    print("Median radius of all reals:         ", real_radii.median().item())
    print("Median radius of those reals:       ", real_radii[culprits].median().item())

    top = swallowed.argsort(descending=True)[:5]
    print("Top 5 reals: noise swallowed", swallowed[top].tolist())
    print("             their radii    ", [round(r, 2) for r in real_radii[top].tolist()])
    print("             their classes  ", real_labels[top].tolist())

    # Shape of the radius distributions, not just the median
    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    print("Real radius quantiles: ", [round(v, 2) for v in torch.quantile(real_radii, qs).tolist()])
    print("Noise radius quantiles:", [round(v, 2) for v in torch.quantile(noise_radii, qs).tolist()])

    # How does precision split over the labels the judge gives to noise?
    hist = torch.bincount(noise_labels, minlength=NUM_CLASSES)
    print("Noise label histogram:", hist.tolist())
    covered = inside.any(dim=0)  # per noise: inside any real ball?
    for c in hist.argsort(descending=True)[:2].tolist():
        mask = noise_labels == c
        print(f"  class {c}: n = {mask.sum().item()}, "
              f"precision = {covered[mask].float().mean().item():.4f}")


def main():
    ref = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")
    feat_a, feat_b = ref["feat_a"], ref["feat_b"]
    judge = load_judge(CHECKPOINT_DIR / "judge_fmnist.pt", device="cpu")

    labels_a = judge.head(feat_a).argmax(dim=1)
    labels_b = judge.head(feat_b).argmax(dim=1)

    results = {}

    # Scenario 1: B
    results["A-B"] = evaluate(feat_a, feat_b, labels_b)

    # Scenario 2: noise
    x_noise = noise_images(len(feat_a), NOISE_SEED)
    noise_features = extract_features(judge, x_noise)
    noise_labels = judge.head(noise_features).argmax(dim=1)
    results["noise"] = evaluate(feat_a, noise_features, noise_labels)

    # Scenario 3: Dropped Class
    data = load_fmnist()
    x_test, y_test = data["x_test"], data["y_test"]
    idx_a, idx_b = ref["idx_a"], ref["idx_b"]
    assert torch.equal(torch.bincount(y_test[idx_b], minlength=NUM_CLASSES),
                       torch.full((NUM_CLASSES,), 300)), "idx_b does not match the stratified split"

    feat_drop = drop_class_set(feat_b, idx_b, idx_a, x_test, y_test, DROPPED_CLASS, judge, DROP_SEED, NUM_CLASSES)
    labels_drop = judge.head(feat_drop).argmax(dim=1)
    results[f"drop {DROPPED_CLASS}"] = evaluate(feat_a, feat_drop, labels_drop)

    print_table(results)
    drop_class_sweep(feat_a, feat_b, idx_a, idx_b, x_test, y_test, judge, results["A-B"])

    # Reference-only runs used by the checks (not printed in the table)
    a_vs_a = evaluate(feat_a, feat_a, labels_a)
    b_vs_a = evaluate(feat_b, feat_a, labels_a)

    print("Sanity checks\n-------------")
    # Identity and symmetry
    check(all(a_vs_a[m] == 1.0 for m in ("coverage", "density", "precision", "recall")),
          "A vs A: all PRDC metrics are exactly 1")
    check(results["A-B"]["precision"] == b_vs_a["recall"], "precision(A, B) == recall(B, A)")
    check(b_vs_a["precision"] == results["A-B"]["recall"], "precision(B, A) == recall(A, B)")

    # Noise must collapse the metrics that theory says it should
    check(results["noise"]["coverage"] < 0.05, "noise: coverage < 0.05")
    check(results["noise"]["recall"] < 0.05, "noise: recall < 0.05")
    check(results["noise"]["density"] < 0.5 * results["A-B"]["density"],
          "noise: density < half of baseline")
    check(results["noise"]["fid"] > 10 * results["A-B"]["fid"], "noise: FID > 10 x baseline")

    # Entropy bound
    check(results["A-B"]["class_entropy"] <= math.log(NUM_CLASSES),
          "A-B: class entropy <= log(NUM_CLASSES)")

    # Dropped class: diversity metrics fall, fidelity does not
    drop = results[f"drop {DROPPED_CLASS}"]
    base = results["A-B"]
    check((labels_b == y_test[idx_b]).float().mean() > 0.9,
          "feat_b is aligned with idx_b (judge accuracy on B > 0.9)")
    check(drop["coverage"] < base["coverage"] - 0.05, "drop: coverage falls by > 0.05")
    check(drop["recall"] < base["recall"] - 0.05, "drop: recall falls by > 0.05")
    check(drop["class_entropy"] < base["class_entropy"] - 0.05, "drop: entropy falls by > 0.05")
    check(abs(drop["precision"] - base["precision"]) < 0.02, "drop: precision unchanged (within 0.02)")

    print(f"\n[NOTE] noise precision = {results['noise']['precision']:.4f} while "
          f"density = {results['noise']['density']:.4f}: precision counts a dense fake "
          "cloud near a few real balls as real (Naeem et al. 2020), so it is not "
          "used as a collapse check.")


if __name__ == "__main__":
    main()

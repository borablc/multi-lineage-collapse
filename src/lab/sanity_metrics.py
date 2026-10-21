"""Sanity checks for the metric module on fake sets with known answers.

Scenarios (real reference is always set A, 3000 samples):
  1. B:     a second real set from the same distribution (perfect generator)
  2. noise: uniform random pixels (generator that produces garbage)
  3. (next step) B with one class removed (generator that forgot a mode)
"""
import math

import torch

from lab.judge import NUM_CLASSES, extract_features, load_judge
from lab.metrics import (DEFAULT_K, _inside, class_entropy, fid, knn_radii,
                         pairwise_distances, prdc)
from lab.paths import CHECKPOINT_DIR

NOISE_SEED = 9001  # IT'S OVER 9000!


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


def print_table(results: dict[str, dict[str, float]]) -> None:
    """One block per scenario, one line per metric, 4 decimals."""
    for scenario, metrics in results.items():
        print(f"-------------------- Scenario: {scenario} --------------------")
        for metric, value in metrics.items():
            print(f" {metric:<14}: {value:>10.4f}")
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

    print_table(results)

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

    print(f"\n[NOTE] noise precision = {results['noise']['precision']:.4f} while "
          f"density = {results['noise']['density']:.4f}: precision counts a dense fake "
          "cloud near a few real balls as real (Naeem et al. 2020), so it is not "
          "used as a collapse check.")


if __name__ == "__main__":
    main()
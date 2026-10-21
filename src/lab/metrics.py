"""Sample-quality metrics computed on frozen judge features.

Part 1: k-NN family (Precision, Recall, Density, Coverage).
References: Kynkaanniemi et al. 2019 (P/R), Naeem et al. 2020 (D/C).

Part 2: FID.
Reference: Heusel et al. 2017.

Part 3: Class Entropy.
"""
import math

import torch

from lab.judge import load_judge, NUM_CLASSES
from lab.paths import CHECKPOINT_DIR

DEFAULT_K = 5


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(N, D), (M, D) -> (N, M) Euclidean distances."""
    return torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')


def knn_radii(feat: torch.Tensor, k: int) -> torch.Tensor:
    """(N, D) -> (N,) distance from each sample to its k-th nearest neighbor
    within the same set, excluding itself."""
    d = pairwise_distances(feat, feat)
    return torch.kthvalue(d, k + 1, dim=1).values


def _inside(radii: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(N_ref,), (N_ref, N_query) -> (N_ref, N_query) bool.

    Entry [i, j] is True if query sample j lies strictly inside the k-NN ball
    of reference sample i. Works in both directions: real balls with d,
    fake balls with d.T.
    """
    return d < radii[:, None]


def _coverage(real_radii: torch.Tensor, d: torch.Tensor) -> float:
    """Fraction of real samples whose nearest fake sample lies inside their k-NN ball."""
    closest = d.min(dim=1).values
    covered = closest < real_radii
    return covered.float().mean().item()


def _precision(real_radii: torch.Tensor, d: torch.Tensor) -> float:
    """Fraction of fake samples that lie inside at least one real k-NN ball."""
    inside = _inside(real_radii, d)
    return inside.any(dim=0).float().mean().item()


def _density(real_radii: torch.Tensor, d: torch.Tensor, k: int) -> float:
    """Average number of real k-NN balls containing each fake sample, divided by k."""
    inside = _inside(real_radii, d)
    return inside.float().sum().item() / (k * d.shape[1])


def prdc(real_feat: torch.Tensor, fake_feat: torch.Tensor, k: int = DEFAULT_K) -> dict[str, float]:
    """(N_real, D), (N_fake, D) -> results_dict.
    Returns the values of coverage, density, precision and recall metrics for a given set of real and fake samples.

    Args:
        real_feat (torch.Tensor): Feature representations of the real dataset samples.
        fake_feat (torch.Tensor): Feature representations of the fake/generated dataset samples.
        k (int, optional): Number of nearest neighbors to consider. Defaults to DEFAULT_K.

    Returns:
        dict[str, float]: A dictionary containing the following metric scores:

            - "coverage": Fraction of real samples whose nearest fake sample lies inside their k-NN ball.
            - "density": Average number of real k-NN balls containing each fake sample, divided by k.
            - "precision": Fraction of fake samples that lie inside at least one real k-NN ball.
            - "recall": Fraction of real samples that lie inside at least one fake k-NN ball.
     """
    # Common values
    real_radii = knn_radii(real_feat, k)
    fake_radii = knn_radii(fake_feat, k)
    d = pairwise_distances(real_feat, fake_feat)

    # Metrics
    coverage = _coverage(real_radii, d)
    density = _density(real_radii, d, k)
    precision = _precision(real_radii, d)
    recall = _precision(fake_radii, d.T)

    return {"coverage": coverage, "density": density, "precision": precision, "recall": recall}


def gaussian_stats(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, D) -> (mean (D,), cov (D, D)), both float64."""
    feat = feat.double()
    D = feat.size(1)
    mu = feat.mean(dim=0)
    cov = torch.cov(feat.T).reshape(D, D)
    return mu, cov


def _sqrt_psd(mat: torch.Tensor) -> torch.Tensor:
    """Square root of a symmetric positive semi-definite matrix. (D, D) -> (D, D)."""
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp(min=0)
    return vecs @ torch.diag(vals.sqrt()) @ vecs.T


def fid_from_stats(real_mu: torch.Tensor, real_cov: torch.Tensor,
                   fake_mu: torch.Tensor, fake_cov: torch.Tensor) -> float:
    """Frechet distance between two Gaussians given their statistics."""

    # 1. Calculating the mu difference term (||mu_r - mu_f||^2)
    mu_diff_term = torch.sum((real_mu - fake_mu) ** 2)

    # 2. Calculating the trace term ( Tr(cov_r + cov_f - 2 * (cov_r^(1/2) @ cov_f @ cov_r^(1/2))^(1/2)) )
    # 2.1. Calculating the cov_r^(1/2) term
    real_cov_sqrt = _sqrt_psd(real_cov)
    # 2.2. Calculating the cov_r^(1/2) @ cov_f @ cov_r^(1/2) term
    sandwich = real_cov_sqrt @ fake_cov @ real_cov_sqrt
    # 2.3. Calculating the (cov_r^(1/2) @ cov_f @ cov_r^(1/2))^(1/2)) term
    sandwich_sqrt = _sqrt_psd(sandwich)
    # 2.4. Calculating the full trace term
    trace_term = torch.trace(real_cov + fake_cov - 2 * sandwich_sqrt)

    # 3. Calculate the FID
    return (mu_diff_term + trace_term).item()


def fid(real_feat: torch.Tensor, fake_feat: torch.Tensor) -> float:
    """Frechet distance between Gaussians fitted to two feature sets.

    FID = ||mu_r - mu_f||^2 + Tr(cov_r + cov_f - 2 * (cov_r @ cov_f)^(1/2))

    The matrix square root is computed in the symmetric form
        Tr((cov_r @ cov_f)^(1/2)) = Tr((S @ cov_f @ S)^(1/2)),  S = cov_r^(1/2)
    because S @ cov_f @ S is symmetric PSD, so eigh can be used and no complex
    values appear.

    In 1-D this reduces to (mu_r - mu_f)^2 + (sigma_r - sigma_f)^2.
    """
    real_mu, real_cov = gaussian_stats(real_feat)
    fake_mu, fake_cov = gaussian_stats(fake_feat)

    return fid_from_stats(real_mu, real_cov, fake_mu, fake_cov)


def check(condition: bool, description: str) -> None:
    """Assert a sanity condition and report it by name."""
    assert condition, f"FAILED: {description}"
    print(f"[PASSED] {description}")


def class_entropy(labels: torch.Tensor, num_classes: int) -> float:
    """(N,) int64 predicted labels -> Shannon entropy of the label histogram (natural log).

    Max is log(num_classes) (perfectly balanced), min is 0 (all one class).
    Blind to within-class collapse: one sample per class still scores the max.
    """
    counts = torch.bincount(labels, minlength=num_classes)
    p = counts / counts.sum()
    H = -torch.sum(torch.special.xlogy(p, p)).item()
    return H


def main():
    ref = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")
    feat_a, feat_b = ref["feat_a"], ref["feat_b"]

    prdc_aa = prdc(feat_a, feat_a, k=DEFAULT_K)
    prdc_ab = prdc(feat_a, feat_b, k=DEFAULT_K)
    prdc_ba = prdc(feat_b, feat_a, k=DEFAULT_K)

    # PRDC
    print("PRDC values\n-------------")
    print("A vs A's PRDC:", prdc_aa)
    print("A vs B's PRDC:", prdc_ab)
    print("B vs A's PRDC:", prdc_ba)

    # PRDC Sanity Checks
    print("\nPRDC sanity checks\n-------------")
    check(all(v == 1.0 for v in prdc_aa.values()), "A vs A: all PRDC metrics are exactly 1")
    check(prdc_ab["precision"] == prdc_ba["recall"], "precision(A, B) == recall(B, A)")
    check(prdc_ba["precision"] == prdc_ab["recall"], "precision(B, A) == recall(A, B)")

    # FID
    fid_aa = fid(feat_a, feat_a)
    fid_ab = fid(feat_a, feat_b)
    fid_ba = fid(feat_b, feat_a)
    print("\nFID values\n-------------")
    print("A vs A's FID:", fid_aa)
    print("A vs B's FID:", fid_ab)
    print("B vs A's FID:", fid_ba)

    # FID Sanity checks
    print("\nFID sanity checks\n-------------")
    check(abs(fid_aa) < 1e-6, "A vs A: FID is ~0")
    check(math.isclose(fid_ab, fid_ba, rel_tol=1e-6), "FID(A, B) ~= FID(B, A)")

    # Class Entropy
    judge = load_judge(CHECKPOINT_DIR / "judge_fmnist.pt", device="cpu")
    labels_a = judge.head(feat_a).argmax(dim=1)
    labels_b = judge.head(feat_b).argmax(dim=1)

    print("\nClass Entropy values\n-------------")
    class_entropy_a = class_entropy(labels_a, NUM_CLASSES)
    class_entropy_b = class_entropy(labels_b, NUM_CLASSES)
    print("Class Entropy feat A:", class_entropy_a)
    print("Class Entropy feat B:", class_entropy_b)


if __name__ == "__main__":
    main()

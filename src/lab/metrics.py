"""Sample-quality metrics computed on frozen judge features.

Part 1: k-NN family (Precision, Recall, Density, Coverage).
References: Kynkaanniemi et al. 2019 (P/R), Naeem et al. 2020 (D/C).
"""
import torch

from lab.paths import CHECKPOINT_DIR

DEFAULT_K = 5


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(N, D), (M, D) -> (N, M) Euclidean distances."""
    return torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')


def knn_radii(feat: torch.Tensor, k: int) -> torch.Tensor:
    """(N, D) -> (N,) distance from each sample to its k-th nearest neighbor
    within the same set, excluding itself."""
    d = pairwise_distances(feat, feat)
    return torch.kthvalue(d, k+1, dim=1).values


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


def main():
    ref = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")
    feat_a, feat_b, feat_train = ref["feat_a"], ref["feat_b"], ref["feat_train"]

    a_vs_a = prdc(feat_a, feat_a, k=DEFAULT_K)
    a_vs_b = prdc(feat_a, feat_b, k=DEFAULT_K)
    b_vs_a = prdc(feat_b, feat_a, k=DEFAULT_K)

    print("A vs A:",a_vs_a)
    print("A vs B:",a_vs_b)
    print("B vs A:",b_vs_a)

    # Sanity Checks
    assert all(v == 1.0 for v in a_vs_a.values()), a_vs_a
    assert a_vs_b["precision"] == b_vs_a["recall"]
    assert b_vs_a["precision"] == a_vs_b["recall"]

    print("\nSanity checks\n-------------")
    print("In A vs A, all metrics must be 1:", a_vs_a["coverage"] == a_vs_a["density"] == a_vs_a["precision"] == a_vs_a["recall"] == 1)
    print("A vs B's precision must be the same as B vs A's recall:", a_vs_b["precision"] == b_vs_a["recall"])
    print("B vs A's precision must be the same as A vs B's recall:", b_vs_a["precision"] == a_vs_b["recall"])


if __name__ == "__main__":
    main()
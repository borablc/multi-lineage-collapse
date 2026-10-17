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

def coverage(real_feat: torch.Tensor, fake_feat: torch.Tensor, k: int = DEFAULT_K) -> float:
    """Fraction of real samples whose nearest fake sample lies inside their k-NN ball."""
    real_radii = knn_radii(real_feat, k)
    d = pairwise_distances(real_feat, fake_feat)
    closest = d.min(dim=1).values
    results = closest < real_radii
    return results.float().mean().item()

def main():
    ref = torch.load(CHECKPOINT_DIR / "reference_fmnist.pt")
    feat_a, feat_b, feat_train = ref["feat_a"], ref["feat_b"], ref["feat_train"]

    print(coverage(feat_a, feat_a)) # Sanity check, must be exactly 1.0
    print(coverage(feat_a, feat_b)) # Coverage baseline, must be around .95-.97
    print(coverage(feat_b, feat_a)) # Coverage is not symmetrical, but expected to be close to a-b.


if __name__ == "__main__":
    main()
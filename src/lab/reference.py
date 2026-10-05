"""Builds the fixed real reference subsamples and caches their judge features. Run once."""
import torch

from lab.data import load_fmnist
from lab.judge import extract_features, load_judge
from lab.paths import CHECKPOINT_DIR

SEED = 1234
PER_CLASS = 300
NUM_CLASSES = 10


def stratified_split(y: torch.Tensor, per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Picks two disjoint index sets, each with `per_class` samples of every class."""
    g = torch.Generator().manual_seed(seed)  # local RNG, independent of all other randomness
    idx_a, idx_b = [], []
    for c in range(NUM_CLASSES):
        class_idx = torch.nonzero(y == c).squeeze(1)
        class_idx = class_idx[torch.randperm(len(class_idx), generator=g)]
        idx_a.append(class_idx[:per_class])
        idx_b.append(class_idx[per_class:2 * per_class])
    return torch.cat(idx_a), torch.cat(idx_b)


def main() -> None:
    device = "cuda"
    d = load_fmnist(device="cpu")  # index selection happens on CPU
    judge = load_judge(CHECKPOINT_DIR / "judge_fmnist.pt", device)

    idx_a, idx_b = stratified_split(d["y_test"], PER_CLASS, SEED)

    feat_a = extract_features(judge, d["x_test"][idx_a].to(device))
    feat_b = extract_features(judge, d["x_test"][idx_b].to(device))
    feat_train = extract_features(judge, d["x_train"].to(device))

    ref = {"seed": SEED, "idx_a": idx_a, "idx_b": idx_b,
           "feat_a": feat_a, "feat_b": feat_b, "feat_train": feat_train}
    torch.save(ref, CHECKPOINT_DIR / "reference_fmnist.pt")

    # --- sanity checks ---
    print("A:", tuple(feat_a.shape), " B:", tuple(feat_b.shape), " train:", tuple(feat_train.shape))
    print("A class counts:", torch.bincount(d["y_test"][idx_a]).tolist())
    print("B class counts:", torch.bincount(d["y_test"][idx_b]).tolist())
    print("A/B overlap:", len(set(idx_a.tolist()) & set(idx_b.tolist())))
    std = feat_train.std(dim=0)
    print(f"feature std: min {std.min():.3f}  max {std.max():.3f}  dead dims {(std < 1e-3).sum().item()}")


if __name__ == "__main__":
    main()
from pathlib import Path

import torch
from torchvision import datasets

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def load_fmnist(device: str = "cpu") -> dict[str, torch.Tensor]:
    """Returns raw FashionMNIST dataset.

    Returned Dict:
        x_train: (60000, 28, 28) uint8
        y_train: (60000,)        int64
        x_test:  (10000, 28, 28) uint8
        y_test:  (10000,)        int64
    """
    ds_train = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True)
    ds_test = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True)
    ds_dict = {"x_train": ds_train.data.to(device), "y_train": ds_train.targets.to(device),
               "x_test": ds_test.data.to(device), "y_test": ds_test.targets.to(device)}
    return ds_dict


def preprocess(x_uint8: torch.Tensor) -> torch.Tensor:
    """Transforms uint8 to model input.

    (N, 28, 28) uint8, 0..255 -> (N, 1, 28, 28) float32, 0..1
    """
    assert x_uint8.dtype == torch.uint8, f"preprocess expects uint8, got {x_uint8.dtype}"
    x_float = x_uint8.float() / 255
    x_float = x_float.unsqueeze(1)
    return x_float


if __name__ == "__main__":
    d = load_fmnist()
    for k, v in d.items():
        print(k, tuple(v.shape), v.dtype)
    print("Class counts:", torch.bincount(d["y_train"]).tolist())
    x = preprocess(d["x_train"][:5])
    print("preprocess:", tuple(x.shape), x.dtype, float(x.min()), float(x.max()))
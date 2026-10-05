"""Frozen judge network: feature extractor and class labeler used by all metrics."""
import torch
from torch import nn

FEATURE_DIM = 128
NUM_CLASSES = 10


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 conv layers with BatchNorm and ReLU, followed by 2x2 max pooling.

    Halves the spatial size: (N, in_ch, H, W) -> (N, out_ch, H/2, W/2)
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        nn.MaxPool2d(2)
    )


class JudgeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(conv_block(1, 32), conv_block(32, 64), nn.Flatten())
        self.bottleneck = nn.Linear(64 * 7 * 7, FEATURE_DIM)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.3), nn.Linear(FEATURE_DIM, NUM_CLASSES))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, 28, 28) float -> (N, FEATURE_DIM). Bottleneck output, taken before the ReLU."""
        return self.bottleneck(self.backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 1, 28, 28) float -> (N, NUM_CLASSES) logits."""
        return self.head(self.features(x))


def load_judge(path, device: str = "cuda") -> JudgeCNN:
    """Loads the trained judge in eval mode with all gradients disabled (frozen)."""
    model = JudgeCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


if __name__ == "__main__":
    model = JudgeCNN()
    x = torch.rand(4, 1, 28, 28)
    print("features:", tuple(model.features(x).shape))
    print("logits:  ", tuple(model(x).shape))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
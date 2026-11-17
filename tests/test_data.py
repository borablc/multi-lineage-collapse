"""Unit tests for lab.data on hand-computed toy cases."""
import torch

from lab.data import preprocess, quantize


def test_round_trip():
    torch.manual_seed(0)

    guaranteed = torch.arange(0, 256, dtype=torch.uint8)
    remaining_count = (3 * 1 * 30 * 20) - 256
    extras = torch.randint(0, 256, (remaining_count,), dtype=torch.uint8)
    tensor_1d = torch.cat([guaranteed, extras])
    tensor_1d = tensor_1d[torch.randperm(tensor_1d.numel())]
    x = tensor_1d.view(3, 30, 20)

    out = quantize(preprocess(x))
    assert torch.equal(out, x)


def test_quantize_shape():
    x = torch.rand(3, 1, 28, 28)
    out = quantize(x)

    assert out.shape == (3, 28, 28)
    assert out.dtype == torch.uint8


def test_quantize_rounding():
    x = torch.full((1, 1, 1, 1), 0.3015)  # 0.3015 * 255 = 76.8825

    expected = 77
    out = quantize(x)

    assert out.item() == expected


def test_quantize_clamp():
    x = torch.tensor([[[[-0.1, 1.2]]]])

    expected = torch.tensor([[[0, 255]]], dtype=torch.uint8)
    out = quantize(x)

    assert torch.equal(out, expected)

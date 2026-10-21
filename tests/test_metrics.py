"""Unit tests for lab.metrics on hand-computed toy cases."""
from math import log

import pytest
import torch

from lab.metrics import _sqrt_psd, fid, prdc, class_entropy

F64 = torch.float64
I64 = torch.int64

# ---------------------------------------------------------------------------
# k-NN family: the "houses on a street" toy (1-D, k = 2)
# Real radii: [3, 2, 2, 3, 7]
HOUSES_REAL = torch.tensor([[0.], [1.], [3.], [4.], [10.]])
HOUSES_FAKE = torch.tensor([[0.5], [1.2], [2.0], [-6.]])


def test_houses_prdc():
    res = prdc(HOUSES_REAL, HOUSES_FAKE, k=2)
    assert res["coverage"] == pytest.approx(0.8)
    assert res["precision"] == pytest.approx(0.75)
    assert res["density"] == pytest.approx(1.25)
    assert res["recall"] == pytest.approx(0.6)


def test_identity_gives_all_ones():
    res = prdc(HOUSES_REAL, HOUSES_REAL, k=2)
    assert all(v == pytest.approx(1.0) for v in res.values())


def test_precision_recall_swap():
    res_rf = prdc(HOUSES_REAL, HOUSES_FAKE, k=2)
    res_fr = prdc(HOUSES_FAKE, HOUSES_REAL, k=2)

    assert res_rf["precision"] == pytest.approx(res_fr["recall"])
    assert res_fr["precision"] == pytest.approx(res_rf["recall"])


# ---------------------------------------------------------------------------
# Matrix square root
def test_sqrt_psd_diagonal():
    example_torch = torch.diag(torch.tensor([4., 9.], dtype=F64))
    example_torch_sqrt = _sqrt_psd(example_torch)
    assert torch.allclose(example_torch_sqrt, torch.diag(torch.tensor([2., 3], dtype=F64)))


def test_sqrt_psd_squares_back():
    torch.manual_seed(73)
    rnd_tensor = torch.rand(3, 3, dtype=F64)
    rnd_tensor = rnd_tensor @ rnd_tensor.T
    assert torch.allclose(rnd_tensor, _sqrt_psd(rnd_tensor) @ _sqrt_psd(rnd_tensor))


# ---------------------------------------------------------------------------
# FID: 3-D toys
REAL_3D = torch.tensor([[3., 5., 7.],
                        [3., 3., 5.],
                        [1., 5., 5.],
                        [1., 3., 7.]], dtype=F64)

FAKE_3D_SHIFTED = torch.tensor([[6., 9., 12.],
                                [6., 7., 10.],
                                [4., 9., 10.],
                                [4., 7., 12.]], dtype=F64)

FAKE_3D_SHIFTED_WIDER = torch.tensor([[7., 10., 13.],
                                      [7., 6., 9.],
                                      [3., 10., 9.],
                                      [3., 6., 13.]], dtype=F64)

FAKE_3D_WIDER = torch.tensor([[4., 6., 8.],
                              [4., 2., 4.],
                              [0., 6., 4.],
                              [0., 2., 8.]], dtype=F64)

FAKE_3D_COLLAPSED = torch.tensor([[2., 4., 6.],
                                  [2., 4., 6.],
                                  [2., 4., 6.],
                                  [2., 4., 6.]], dtype=F64)

FAKE_3D_TILTED = torch.tensor([[4., 5., 7.],
                               [2., 3., 5.],
                               [2., 5., 5.],
                               [0., 3., 7.]], dtype=F64)


@pytest.mark.parametrize("fake, expected", [
    (FAKE_3D_SHIFTED, 50.),
    (FAKE_3D_SHIFTED_WIDER, 54.),
    (FAKE_3D_WIDER, 4.),
    (FAKE_3D_COLLAPSED, 4.),
    (FAKE_3D_TILTED, .703818726667),
], ids=["shifted", "shifted_wider", "wider", "collapsed", "tilted"])
def test_fid_3d(fake, expected):
    assert fid(REAL_3D, fake) == pytest.approx(expected)


# FID: 1-D toys (regression to the scalar formula)
REAL_1D = torch.tensor([1., 2., 3.], dtype=F64)[:, None]
FAKE_1D_SHIFTED = torch.tensor([4., 5., 6.], dtype=F64)[:, None]
FAKE_1D_SHIFTED_WIDER = torch.tensor([3., 5., 7.], dtype=F64)[:, None]
FAKE_1D_WIDER = torch.tensor([0., 2., 4.], dtype=F64)[:, None]
FAKE_1D_COLLAPSED = torch.tensor([2., 2., 2.], dtype=F64)[:, None]


@pytest.mark.parametrize("fake, expected", [
    (FAKE_1D_SHIFTED, 9.),
    (FAKE_1D_SHIFTED_WIDER, 10.),
    (FAKE_1D_WIDER, 1.),
    (FAKE_1D_COLLAPSED, 1.),
], ids=["shifted", "shifted_wider", "wider", "collapsed"])
def test_fid_1d(fake, expected):
    assert fid(REAL_1D, fake) == pytest.approx(expected)


# FID: orthogonal clouds (diagonal-only implementation would give 0)
REAL_YATS = torch.tensor(
    [[-2., -2.],
     [-1., -1.],
     [0., 0.],
     [1., 1.],
     [2., 2.]], dtype=F64)

FAKE_YATS = torch.tensor(
    [[-2., 2.],
     [-1., 1],
     [0., 0.],
     [1., -1.],
     [2., -2]], dtype=F64)


def test_fid_orthogonal_clouds():
    assert fid(REAL_YATS, FAKE_YATS) == pytest.approx(10.)


@pytest.mark.parametrize("fake", [
    FAKE_3D_SHIFTED,
    FAKE_3D_SHIFTED_WIDER,
    FAKE_3D_WIDER,
    FAKE_3D_COLLAPSED,
    FAKE_3D_TILTED,
], ids=["shifted", "shifted_wider", "wider", "collapsed", "tilted"])
def test_fid_symmetry(fake):
    assert fid(REAL_3D, fake) == pytest.approx(fid(fake, REAL_3D))


def test_fid_identity():
    assert fid(REAL_3D, REAL_3D) == pytest.approx(0, abs=1e-9)

# ---------------------------------------------------------------------------
# Class Entropy

@pytest.mark.parametrize("labels, num_classes, expected", [
    (torch.arange(10).repeat(300), 10, log(10)),
    (torch.arange(2).repeat(1500), 10, log(2)),
    (torch.tensor([0, 0, 0, 1], dtype=I64).repeat(750), 10, -(0.75 * log(0.75) + 0.25 * log(0.25))),
    (torch.zeros(3000, dtype=I64), 10, 0.)
], ids=["evenly_distributed", "half_one_half_zero", "three_quarters_zero_quarter_one", "collapsed"])
def test_class_entropy(labels, num_classes, expected):
    assert class_entropy(labels, num_classes) == pytest.approx(expected)
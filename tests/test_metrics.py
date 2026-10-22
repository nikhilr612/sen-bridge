import numpy as np
import pytest

from sen_bridge import fid_core

torch = pytest.importorskip("torch")
from sen_bridge import torch as sbt  # noqa: E402


def test_identity_and_ordering():
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    assert sbt.ssim(x, x).min() > 0.999
    assert sbt.psnr(x, x).min() > 90
    assert (sbt.psnr(x, x * 0.5) < sbt.psnr(x, x * 0.9)).all()


def test_frechet_zero_for_same_stats():
    f = np.random.default_rng(0).normal(size=(500, 8))
    s = fid_core.feature_stats(f)
    assert fid_core.frechet_distance(*s, *s) == pytest.approx(0, abs=1e-6)

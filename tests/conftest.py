import numpy as np
import pytest


def structured_images(n, seed=0):
    """Deterministic [n, 3, 128, 128] float images in [-1, 1]: smoothed noise plus a gradient."""
    rng = np.random.default_rng(seed)
    a = rng.random((n, 3, 128, 128)).astype(np.float32)
    k = np.ones(5, np.float32) / 5
    for ax in (2, 3):
        a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, a)
    g = np.linspace(-1, 1, 128, dtype=np.float32)
    return np.clip(a * 3 - 1 + 0.5 * g[None, None, :, None] * g[None, None, None, :], -1, 1)


@pytest.fixture(scope="session")
def pair():
    x = structured_images(12)
    rng = np.random.default_rng(1)
    return x, np.clip(x + 0.15 * rng.standard_normal(x.shape).astype(np.float32), -1, 1)


@pytest.fixture(scope="session")
def weights_ready():
    from sen_bridge import weights
    try:
        weights.path("fid_inception"), weights.path("lpips_vgg")
    except Exception as e:  # offline or host unavailable
        pytest.skip(f"weights unavailable: {e}")

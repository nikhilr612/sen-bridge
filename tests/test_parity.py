"""PyTorch and JAX give the same numbers: FID within 0.01, LPIPS within 1e-4 (proposal's CI parity test)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from sen_bridge import jax as sbj, torch as sbt  # noqa: E402


def nhwc(a):
    return jnp.asarray(np.transpose(a, (0, 2, 3, 1)))


def test_psnr_ssim_parity(pair):
    x, y = pair
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for name in ("psnr", "ssim"):
        j = np.asarray(jax.jit(jax.vmap(getattr(sbj, name)))(nhwc(x), nhwc(y)))
        assert np.abs(j - getattr(sbt, name)(xt, yt).numpy()).max() < 1e-4


def test_lpips_parity(pair, weights_ready):
    x, y = pair
    j = np.asarray(jax.jit(jax.vmap(sbj.lpips))(nhwc(x), nhwc(y)))
    assert np.abs(j - sbt.lpips(torch.from_numpy(x), torch.from_numpy(y)).numpy()).max() < 1e-4


def test_fid_parity(weights_ready):
    from conftest import structured_images
    real, fake = structured_images(60, seed=3), structured_images(60, seed=4)
    ft = sbt.FID(device="cpu")
    fj = sbj.FID()
    for lo in range(0, 60, 20):
        for acc, conv in ((ft, torch.from_numpy), (fj, nhwc)):
            acc.update(conv(real[lo:lo + 20]), True)
            acc.update(conv(fake[lo:lo + 20]), False)
    assert abs(ft.compute() - fj.compute()) < 0.01


def test_lpips_first_call_inside_jit_does_not_leak_tracers(pair, weights_ready):
    sbj._lpips_params.cache_clear()
    f = jax.jit(sbj.lpips)
    x, y = nhwc(pair[0]), nhwc(pair[1])
    a = f(x[0], y[0])
    b = jax.jit(jax.vmap(sbj.lpips))(x[:2], y[:2])  # a second trace reuses the cached parameters
    assert np.isfinite(float(a)) and np.isfinite(np.asarray(b)).all()

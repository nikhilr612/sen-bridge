# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch and JAX give the same numbers: FID within 0.01 and LPIPS within 1e-4."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from conftest import structured_images
from sen_bridge import jax as sbj
from sen_bridge import torch as sbt

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Pair = tuple[NDArray[np.float32], NDArray[np.float32]]

_FID_BATCH = 20


def nhwc(a: NDArray[np.float32]) -> jax.Array:
    """Convert channel-first images to a channel-last JAX array.

    Args:
        a: Images of shape ``[B, 3, H, W]``.

    Returns:
        The images as a ``[B, H, W, 3]`` array.
    """
    return jnp.asarray(np.transpose(a, (0, 2, 3, 1)))


def test_psnr_ssim_parity(pair: Pair) -> None:
    x, y = pair
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for name in ("psnr", "ssim"):
        got = np.asarray(jax.jit(jax.vmap(getattr(sbj, name)))(nhwc(x), nhwc(y)))
        assert np.abs(got - getattr(sbt, name)(xt, yt).numpy()).max() < 1e-4


@pytest.mark.usefixtures("weights_ready")
def test_lpips_parity(pair: Pair) -> None:
    x, y = pair
    got = np.asarray(jax.jit(jax.vmap(sbj.lpips))(nhwc(x), nhwc(y)))
    expected = sbt.lpips(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    assert np.abs(got - expected).max() < 1e-4


@pytest.mark.usefixtures("weights_ready")
def test_fid_parity() -> None:
    real, fake = structured_images(60, seed=3), structured_images(60, seed=4)
    fid_t, fid_j = sbt.FID(device="cpu"), sbj.FID()
    for lo in range(0, 60, _FID_BATCH):
        sl = slice(lo, lo + _FID_BATCH)
        fid_t.update(torch.from_numpy(real[sl]), real=True)
        fid_t.update(torch.from_numpy(fake[sl]), real=False)
        fid_j.update(nhwc(real[sl]), real=True)
        fid_j.update(nhwc(fake[sl]), real=False)
    assert abs(fid_t.compute() - fid_j.compute()) < 0.01


@pytest.mark.usefixtures("weights_ready")
def test_lpips_first_call_inside_jit_does_not_leak_tracers(pair: Pair) -> None:
    sbj._lpips_params.cache_clear()
    x, y = nhwc(pair[0]), nhwc(pair[1])
    first = jax.jit(sbj.lpips)(x[0], y[0])
    batch = jax.jit(jax.vmap(sbj.lpips))(
        x[:2], y[:2]
    )  # a second trace reuses the cache
    assert np.isfinite(float(first))
    assert np.isfinite(np.asarray(batch)).all()


def test_rounding_agrees_between_backends() -> None:
    rng = np.random.default_rng(0)
    ties = (np.arange(256) + 0.5) / 127.5 - 1  # exact rounding ties
    values = np.concatenate([rng.uniform(-1.2, 1.2, 100_000), ties]).astype(
        np.float32
    )
    levels_t = np.round(sbt.quantise(torch.from_numpy(values)).numpy() * 255)
    levels_j = np.round(np.asarray(sbj.quantise(jnp.asarray(values))) * 255)
    assert np.array_equal(
        levels_t, levels_j
    )  # the float values may differ by 1 ulp

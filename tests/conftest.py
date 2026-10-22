# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: deterministic structured images and the weights guard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SIZE = 128
_BLUR = 5


def structured_images(n: int, seed: int = 0) -> NDArray[np.float32]:
    """Make deterministic images in [-1, 1]: smoothed noise plus a gradient.

    Args:
        n: Number of images.
        seed: Seed of the noise.

    Returns:
        Float32 array of shape ``[n, 3, 128, 128]``.
    """
    rng = np.random.default_rng(seed)
    a = rng.random((n, 3, _SIZE, _SIZE)).astype(np.float32)
    kernel = np.ones(_BLUR, np.float32) / _BLUR
    for axis in (2, 3):
        a = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="same"), axis, a
        )
    g = np.linspace(-1, 1, _SIZE, dtype=np.float32)
    ramp = 0.5 * g[None, None, :, None] * g[None, None, None, :]
    return np.clip(a * 3 - 1 + ramp, -1, 1).astype(np.float32)


@pytest.fixture(scope="session")
def pair() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return a batch of images and a noisy copy of it."""
    x = structured_images(12)
    noise = np.random.default_rng(1).standard_normal(x.shape).astype(np.float32)
    return x, np.clip(x + 0.15 * noise, -1, 1).astype(np.float32)


@pytest.fixture(scope="session")
def weights_ready() -> None:
    """Make sure the metric weights are available, or skip the test."""
    from sen_bridge import weights  # ruff: ignore[import-outside-top-level]

    try:
        weights.path("fid_inception")
        weights.path("lpips_vgg")
    except Exception as exc:  # ruff: ignore[blind-except]  (offline or host unavailable)
        pytest.skip(f"weights unavailable: {exc}")

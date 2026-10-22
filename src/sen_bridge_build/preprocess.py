# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Model-free, fixed transforms: crop, 2x2 average, normalise, quantise."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from sen_bridge_build import constants

if TYPE_CHECKING:
    from numpy.typing import NDArray

_Scalar = TypeVar("_Scalar", bound=np.generic)

_MAX_LEVEL = 255
_HALF_RANGE = 127.5
_POWER_FLOOR = 1e-10
_DB_PER_DECADE = 10.0


def center_crop(
    a: NDArray[_Scalar], size: int = constants.CROP
) -> NDArray[_Scalar]:
    """Crop the last two axes around the centre.

    Args:
        a: Array whose last two axes are height and width.
        size: Side of the square crop.

    Returns:
        A view of the central ``size`` by ``size`` window.
    """
    height, width = a.shape[-2:]
    top, left = (height - size) // 2, (width - size) // 2
    return a[..., top : top + size, left : left + size]


def avg_2x2(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Average non-overlapping 2x2 blocks over the last two axes.

    Args:
        a: Array whose last two axes are height and width, both even.

    Returns:
        Array with both spatial axes halved.
    """
    *lead, height, width = a.shape
    return a.reshape(*lead, height // 2, 2, width // 2, 2).mean(axis=(-3, -1))


def sar_to_db128(vv_vh_db: NDArray[np.floating]) -> NDArray[np.float32]:
    """Crop and average radar 2x2 in linear power, then return to dB.

    This is classical multi-looking.

    Args:
        vv_vh_db: Radar VV and VH in dB, shape ``[2, H, W]``.

    Returns:
        Radar in dB, shape ``[2, 128, 128]``.
    """
    cropped = center_crop(np.asarray(vv_vh_db, dtype=np.float32))
    power = avg_2x2(
        np.power(_DB_PER_DECADE, cropped / _DB_PER_DECADE).astype(np.float32)
    )
    return (_DB_PER_DECADE * np.log10(np.maximum(power, _POWER_FLOOR))).astype(
        np.float32
    )


def sar_to_uint8(db: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Map radar to three 8-bit channels (VV, VH, VV-VH) with frozen clips.

    Args:
        db: Radar VV and VH in dB, shape ``[2, 128, 128]``.

    Returns:
        Array of shape ``[3, 128, 128]`` with values 0 to 255.
    """
    channels = np.stack([db[0], db[1], db[0] - db[1]])
    out = np.empty(channels.shape, np.uint8)
    for i, (low, high) in enumerate(constants.SAR_CLIPS):
        unit = (np.clip(channels[i], low, high) - low) / (high - low)
        out[i] = np.round(unit * _MAX_LEVEL).astype(np.uint8)
    return out


def sar_from_uint8(u8: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Invert :func:`sar_to_uint8` back to dB.

    Args:
        u8: Array of shape ``[3, H, W]``.

    Returns:
        VV, VH and VV-VH in dB, same shape.
    """
    out = np.empty(u8.shape, np.float32)
    for i, (low, high) in enumerate(constants.SAR_CLIPS):
        out[i] = low + u8[i].astype(np.float32) / _MAX_LEVEL * (high - low)
    return out


def opt_reflectance128(
    rgb_dn: NDArray[np.number], offset: float = 0.0
) -> NDArray[np.float32]:
    """Crop and 2x2 average Sentinel-2 L2A digital numbers into reflectance.

    Args:
        rgb_dn: Digital numbers, shape ``[C, H, W]``.
        offset: Processing-baseline offset added before scaling (0 for 2019 to 2021
            scenes).

    Returns:
        Reflectance, shape ``[C, 128, 128]``.
    """
    cropped = center_crop(np.asarray(rgb_dn, dtype=np.float32))
    return avg_2x2(((cropped + offset) / constants.S2_SCALE).astype(np.float32))


def opt_to_uint8(rho: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Map reflectance to 8 bits: clip to [0, 0.3], gamma 1/2.2, scale to 255.

    Args:
        rho: Reflectance of any shape.

    Returns:
        Array of the same shape with values 0 to 255.
    """
    unit = (
        np.clip(rho, 0.0, constants.OPT_MAX) / constants.OPT_MAX
    ) ** constants.OPT_GAMMA
    return np.round(unit * _MAX_LEVEL).astype(np.uint8)


def opt_from_uint8(u8: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Invert :func:`opt_to_uint8` back to reflectance.

    Args:
        u8: 8-bit optical image.

    Returns:
        Reflectance of the same shape.
    """
    unit = u8.astype(np.float32) / _MAX_LEVEL
    return (unit ** (1 / constants.OPT_GAMMA) * constants.OPT_MAX).astype(
        np.float32
    )


def to_model_range(u8: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Map 8-bit images to the model range [-1, 1].

    Args:
        u8: 8-bit image.

    Returns:
        Float image in [-1, 1].
    """
    return u8.astype(np.float32) / _HALF_RANGE - 1.0

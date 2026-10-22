# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""The FID resize: clean-fid's "clean" mode (PIL bicubic, float32, 299x299).

Based on clean-fid (MIT, Copyright (c) 2021 Gaurav Parmar), ``cleanfid/resize.py``.
Values stay float in [0, 255] (no re-quantisation after resizing), matching
clean-fid's default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Side length of the Inception input.
INCEPTION_SIZE = 299
_CHANNELS = 3
_MAX_LEVEL = 255


def clean_resize(
    img_u8: NDArray[np.uint8],
    size: int = INCEPTION_SIZE,
) -> NDArray[np.float32]:
    """Resize one image with PIL bicubic interpolation on float32 channels.

    Args:
        img_u8: Image of shape ``[H, W, 3]``.
        size: Output side length.

    Returns:
        Float image of shape ``[size, size, 3]`` with values in [0, 255].
    """
    channels = []
    for c in range(_CHANNELS):
        plane = Image.fromarray(img_u8[:, :, c].astype(np.float32), mode="F")
        resized = plane.resize((size, size), resample=Image.Resampling.BICUBIC)
        channels.append(
            np.asarray(resized).clip(0, _MAX_LEVEL).reshape(size, size, 1)
        )
    return np.concatenate(channels, axis=2).astype(np.float32)


def resize_batch(
    batch_u8: NDArray[np.uint8],
    size: int = INCEPTION_SIZE,
) -> NDArray[np.float32]:
    """Resize a batch of images with :func:`clean_resize`.

    Args:
        batch_u8: Images of shape ``[B, H, W, 3]``.
        size: Output side length.

    Returns:
        Float images of shape ``[B, size, size, 3]`` with values in [0, 255].
    """
    return np.stack([clean_resize(x, size) for x in batch_u8])

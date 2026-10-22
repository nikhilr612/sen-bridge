"""The FID resize: clean-fid's "clean" mode, CPU PIL bicubic on float32 channels to 299x299.

Based on clean-fid (MIT, Copyright (c) 2021 Gaurav Parmar), cleanfid/resize.py. Values stay float in [0, 255]
(no re-quantisation after resizing), matching clean-fid's default.
"""
import numpy as np
from PIL import Image


def clean_resize(img_u8: np.ndarray, size: int = 299) -> np.ndarray:
    """[H, W, 3] uint8 -> [size, size, 3] float32 in [0, 255]."""
    chans = []
    for c in range(3):
        im = Image.fromarray(img_u8[:, :, c].astype(np.float32), mode="F").resize((size, size), resample=Image.BICUBIC)
        chans.append(np.asarray(im).clip(0, 255).reshape(size, size, 1))
    return np.concatenate(chans, axis=2).astype(np.float32)


def resize_batch(batch_u8: np.ndarray, size: int = 299) -> np.ndarray:
    """[B, H, W, 3] uint8 -> [B, size, size, 3] float32 in [0, 255]."""
    return np.stack([clean_resize(x, size) for x in batch_u8])

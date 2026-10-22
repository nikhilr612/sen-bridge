"""Model-free, fixed transforms: crop, 2x2 average, normalise, quantise."""
import numpy as np

from . import constants as C


def _center_crop(a: np.ndarray, size: int = C.CROP) -> np.ndarray:
    h, w = a.shape[-2:]
    t, l = (h - size) // 2, (w - size) // 2
    return a[..., t:t + size, l:l + size]


def _avg2(a: np.ndarray) -> np.ndarray:
    *lead, h, w = a.shape
    return a.reshape(*lead, h // 2, 2, w // 2, 2).mean(axis=(-3, -1))


def sar_to_db128(vv_vh_db: np.ndarray) -> np.ndarray:
    """[2,H,W] dB -> [2,128,128] dB, averaging in linear power (multi-looking)."""
    x = _center_crop(np.asarray(vv_vh_db, dtype=np.float32))
    power = _avg2(np.power(10.0, x / 10.0))
    return 10.0 * np.log10(np.maximum(power, 1e-10))


def sar_to_uint8(db: np.ndarray) -> np.ndarray:
    """[2,128,128] dB -> [3,128,128] uint8 (VV, VH, VV-VH)."""
    ch = np.stack([db[0], db[1], db[0] - db[1]])
    out = np.empty(ch.shape, np.uint8)
    for i, (lo, hi) in enumerate(C.SAR_CLIPS):
        u = (np.clip(ch[i], lo, hi) - lo) / (hi - lo)
        out[i] = np.round(u * 255).astype(np.uint8)
    return out


def sar_from_uint8(u8: np.ndarray) -> np.ndarray:
    """Inverse mapping: [3,H,W] uint8 -> dB channels (VV, VH, VV-VH)."""
    out = np.empty(u8.shape, np.float32)
    for i, (lo, hi) in enumerate(C.SAR_CLIPS):
        out[i] = lo + u8[i].astype(np.float32) / 255 * (hi - lo)
    return out


def opt_reflectance128(rgb_dn: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """[3,H,W] L2A digital numbers (B4,B3,B2) -> [3,128,128] reflectance."""
    x = _center_crop(np.asarray(rgb_dn, dtype=np.float32))
    return _avg2((x + offset) / C.S2_SCALE)


def opt_to_uint8(rho: np.ndarray) -> np.ndarray:
    u = (np.clip(rho, 0.0, C.OPT_MAX) / C.OPT_MAX) ** C.OPT_GAMMA
    return np.round(u * 255).astype(np.uint8)


def opt_from_uint8(u8: np.ndarray) -> np.ndarray:
    return (u8.astype(np.float32) / 255) ** (1 / C.OPT_GAMMA) * C.OPT_MAX


def to_model_range(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 127.5 - 1.0

"""Quality filters. Inputs only; nothing removes a valid-but-hard target."""
import numpy as np

from . import constants as C


def metadata_ok(dt_hours, lat, month, elev=None) -> np.ndarray:
    """Time gap, snow season at |lat|>40, and optional elevation (vectorised)."""
    dt, lat, month = map(np.asarray, (dt_hours, lat, month))
    ok = np.abs(dt) <= C.MAX_DT_HOURS
    winter_n = np.isin(month, (12, 1, 2, 3)) & (lat > C.SNOW_LAT)
    winter_s = np.isin(month, (6, 7, 8, 9)) & (lat < -C.SNOW_LAT)
    ok &= ~(winter_n | winter_s)
    if elev is not None:
        ok &= np.asarray(elev) <= C.ELEV_MAX_M
    return ok


def invalid_pixels(sar_db: np.ndarray, opt_dn: np.ndarray) -> bool:
    """NaN, zero/no-data in either modality."""
    return bool(
        not np.isfinite(sar_db).all() or not np.isfinite(opt_dn).all()
        or (opt_dn == 0).any()
    )


def saturated(opt_rho: np.ndarray) -> bool:
    return bool((opt_rho >= 1.0).any())


def border_noise(sar_db: np.ndarray) -> bool:
    return bool((sar_db[0] < C.BORDER_DB).mean() > C.BORDER_MAX_FRAC)


def rfi_streaks(sar_db: np.ndarray, z: float = 6.0) -> bool:
    """Row/column streak detector: a row or column mean far from the robust median."""
    for ch in sar_db:
        for axis in (0, 1):
            m = ch.mean(axis=axis)
            med = np.median(m)
            mad = np.median(np.abs(m - med)) * 1.4826 + 1e-6
            if (np.abs(m - med) / mad > z).any() and np.abs(m - med).max() > 3.0:
                return True
    return False


def cloud_frac_scl(scl: np.ndarray) -> float:
    """Fraction of cloud (8,9,10 = med/high prob, cirrus) and shadow (3) in an SCL crop."""
    return float(np.isin(scl, (3, 8, 9, 10)).mean())


def flags(sar_db: np.ndarray, dem: np.ndarray | None = None) -> dict:
    out = {
        "water_frac_sar": float((sar_db[0] < C.WATER_DB).mean()),
        "vh_noise_frac": float((sar_db[1] < C.NOISE_FLOOR_DB).mean()),
    }
    if dem is not None:
        gy, gx = np.gradient(dem.astype(np.float32), 20.0)
        out["slope_std"] = float(np.degrees(np.arctan(np.hypot(gx, gy))).std())
        out["elev_mean"] = float(dem.mean())
    return out


def steep(slope_std: float) -> bool:
    return slope_std > C.SLOPE_STD_MAX

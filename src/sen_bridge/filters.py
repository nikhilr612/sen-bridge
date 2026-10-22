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
    return bool((opt_rho >= 1.0).mean() > C.SAT_MAX_FRAC)


def optical_stats(opt_u8: np.ndarray) -> dict:
    """Brightness/flatness of the stored 8-bit RGB [3,H,W]. `flat_bright` is analysis-only:
    it also fires on sand desert, a valid target, so it must never select examples."""
    a = opt_u8.astype(np.float32) / 255
    lum, mn = a.mean(0), a.min(0)
    return {"opt_mean": float(lum.mean()), "opt_std": float(lum.std()), "opt_white_frac": float((mn > 0.85).mean()),
            "flat_bright": bool(lum.std() < 0.04 and mn.mean() > 0.5)}


def border_noise(sar_db: np.ndarray) -> bool:
    return bool((sar_db[0] < C.BORDER_DB).mean() > C.BORDER_MAX_FRAC)


def rfi_score(sar_db: np.ndarray) -> float:
    """Largest robust z-score of a row/column mean over both polarisations (needs a >3 dB offset)."""
    worst = 0.0
    for ch in sar_db:
        for axis in (0, 1):
            m = ch.mean(axis=axis)
            med = np.median(m)
            mad = np.median(np.abs(m - med)) * 1.4826 + 1e-6
            if np.abs(m - med).max() > 3.0:
                worst = max(worst, float((np.abs(m - med) / mad).max()))
    return worst


def rfi_streaks(sar_db: np.ndarray, z: float = 6.0) -> bool:
    return rfi_score(sar_db) > z


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


B01, B02, B03, B04, B8A_, B11 = 0, 1, 2, 3, 8, 10  # indices into the 12-band L2A stack


def spectral_scores(rho12: np.ndarray) -> dict:
    """Fractions on a [12,H,W] reflectance crop; rules fixed before freeze.

    cloud_rule_frac: bright, flat-spectrum pixels (not snow). Also fires on roofs and pale sand,
    so it is a score, not a filter, except for near-total cover.
    snow:  bright green and NDSI > 0.4.
    haze:  mean B01/B02 reflectance ratio; aerosol and thin cloud lift the coastal band.
    """
    g, sw = rho12[B03], rho12[B11]
    ndsi = (g - sw) / np.maximum(g + sw, 1e-6)
    blue, red = rho12[B02], rho12[B04]
    snow = (ndsi > 0.4) & (g > 0.3)
    cloud = (blue > 0.15) & (red > 0.15) & (np.abs(red - blue) < 0.08) & ~snow
    return {
        "cloud_rule_frac": float(cloud.mean()),
        "snow_score": float(snow.mean()),
        "haze_score": float(rho12[B01].mean() / max(blue.mean(), 1e-6)),
    }


SENSEI_CLOUD = (3, 4, 5)  # thin cloud, thick cloud, cloud shadow
SENSEI_NODATA = 6


def sensei_fracs(mask: np.ndarray) -> dict:
    """Fractions from SSL4EO's neural mask on the 128x128 crop; filtering only, never pairing."""
    return {
        "sensei_cloud_frac": float(np.isin(mask, SENSEI_CLOUD).mean()),
        "sensei_nodata_frac": float((mask == SENSEI_NODATA).mean()),
        "water_frac_opt": float((mask == 1).mean()),
        "snow_frac_opt": float((mask == 2).mean()),
    }

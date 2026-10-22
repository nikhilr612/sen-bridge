"""Conditioning vector m, read only from product metadata."""
import numpy as np

from . import constants as C


def meta_vector(doy, lat_deg, lon_deg, dt_hours, orbit_desc) -> np.ndarray:
    """(sin doy, cos doy, x, y, z, dt/48h, orbit +-1) as float32[..., 7]."""
    doy, lat, lon, dt, desc = map(np.asarray, (doy, lat_deg, lon_deg, dt_hours, orbit_desc))
    ang = 2 * np.pi * doy / 365.25
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    v = np.stack([
        np.sin(ang), np.cos(ang),
        np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi),
        dt / C.MAX_DT_HOURS,
        np.where(desc, 1.0, -1.0),
    ], axis=-1)
    return v.astype(np.float32)

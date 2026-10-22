# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Conditioning vector m, read only from product metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sen_bridge_build import constants

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

#: Days per year used for the seasonal encoding.
_DAYS_PER_YEAR = 365.25


def meta_vector(
    doy: ArrayLike,
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    dt_hours: ArrayLike,
    orbit_desc: ArrayLike,
) -> NDArray[np.float32]:
    """Build the conditioning vector for one or many examples.

    The vector is ``(sin doy, cos doy, x, y, z, dt / 48 h, orbit)`` with ``(x, y, z)`` the unit
    vector of the location and the orbit encoded as +1 (descending) or -1 (ascending).

    Args:
        doy: Day of year of the optical acquisition.
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        dt_hours: Optical minus radar acquisition time, in hours.
        orbit_desc: Whether the radar pass is descending.

    Returns:
        Float32 array of shape ``[..., 7]``.
    """
    doy_a, lat, lon, dt = (
        np.asarray(v, dtype=np.float64) for v in (doy, lat_deg, lon_deg, dt_hours)
    )
    desc = np.asarray(orbit_desc, dtype=bool)
    angle = 2 * np.pi * doy_a / _DAYS_PER_YEAR
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    vec = np.stack(
        [
            np.sin(angle),
            np.cos(angle),
            np.cos(phi) * np.cos(lam),
            np.cos(phi) * np.sin(lam),
            np.sin(phi),
            dt / constants.MAX_DT_HOURS,
            np.where(desc, 1.0, -1.0),
        ],
        axis=-1,
    )
    return vec.astype(np.float32)

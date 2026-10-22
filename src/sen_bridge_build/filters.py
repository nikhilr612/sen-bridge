# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Quality filters and per-crop scores.

Every filter uses only information available at prediction time (radar, metadata,
DEM, static maps) or removes ground truth that is invalid; nothing removes a valid
but hard target by looking at it. Scores that did not earn a place as filters
(radio-frequency interference, radar border noise, haze, snow) are recorded but
never used to select examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np

from sen_bridge_build import constants

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

_MAX_LEVEL = 255
_WINTER_NORTH = (12, 1, 2, 3)
_WINTER_SOUTH = (6, 7, 8, 9)
_DEM_PIXEL_M = 20.0
_WHITE_LEVEL = 0.85
_FLAT_STD = 0.04
_FLAT_MIN_CHANNEL = 0.5
_RFI_MIN_OFFSET_DB = 3.0
_MAD_TO_SIGMA = 1.4826
_EPS = 1e-6

# Indices into the 12-band L2A stack (B01, B02, B03, B04, B05, B06, B07, B08, B8A,
# B09, B11, B12).
_B01, _B02, _B03, _B04, _B11 = 0, 1, 2, 3, 10
_NDSI_SNOW = 0.4
_GREEN_SNOW = 0.3
_BRIGHT = 0.15
_WHITE_BALANCE = 0.08

#: SEnSeI mask classes counted as cloud: thin cloud, thick cloud and cloud shadow.
SENSEI_CLOUD = (3, 4, 5)
#: SEnSeI mask class for no data.
SENSEI_NODATA = 6
_SENSEI_WATER = 1
_SENSEI_SNOW = 2

#: Classes of SSL4EO's land-cover layer by code (code 8, clouds, and 0, no data, are
#: not recorded).
LULC_CLASSES = {
    1: "water",
    2: "trees",
    3: "flooded",
    4: "crops",
    5: "built",
    6: "bare",
    7: "snow",
    9: "rangeland",
}


class OpticalStats(TypedDict):
    """Brightness and flatness of the stored 8-bit optical image.

    Attributes:
        opt_mean: Mean luminance in [0, 1].
        opt_std: Standard deviation of the luminance.
        opt_white_frac: Share of near-white pixels (minimum channel above 0.85).
        flat_bright: Featureless and bright. Analysis only: it also fires on sand
            desert, a valid target, so it must never select examples.
    """

    opt_mean: float
    opt_std: float
    opt_white_frac: float
    flat_bright: bool


class SarDemFlags(TypedDict):
    """Radar- and DEM-derived scores of one crop.

    Attributes:
        water_frac_sar: Share of pixels with VV below the water level.
        vh_noise_frac: Share of pixels with VH under the noise floor.
        slope_std: Standard deviation of the terrain slope, in degrees.
        elev_mean: Mean elevation, in metres.
    """

    water_frac_sar: float
    vh_noise_frac: float
    slope_std: float
    elev_mean: float


class SpectralScores(TypedDict):
    """Spectral scores from the 12-band reflectance crop.

    Attributes:
        cloud_rule_frac: Share of bright, flat-spectrum pixels that are not snow. It
            also fires on roofs and pale sand, so it is a score, not a filter,
            except for near-total cover.
        snow_score: Share of bright green pixels with NDSI above 0.4.
        haze_score: Mean B01 over mean B02 reflectance; aerosol and thin cloud lift
            the coastal band.
    """

    cloud_rule_frac: float
    snow_score: float
    haze_score: float


class SenseiFracs(TypedDict):
    """Class shares of SSL4EO's neural mask on the crop.

    Attributes:
        sensei_cloud_frac: Share of cloud and shadow pixels.
        sensei_nodata_frac: Share of no-data pixels.
        water_frac_opt: Share of water pixels.
        snow_frac_opt: Share of snow pixels.
    """

    sensei_cloud_frac: float
    sensei_nodata_frac: float
    water_frac_opt: float
    snow_frac_opt: float


class LulcFracs(TypedDict):
    """Class shares of SSL4EO's land-cover layer on the crop.

    Attributes:
        lulc_water: Share of water.
        lulc_trees: Share of trees.
        lulc_flooded: Share of flooded vegetation.
        lulc_crops: Share of crops.
        lulc_built: Share of built area.
        lulc_bare: Share of bare ground.
        lulc_snow: Share of snow and ice.
        lulc_rangeland: Share of rangeland.
    """

    lulc_water: float
    lulc_trees: float
    lulc_flooded: float
    lulc_crops: float
    lulc_built: float
    lulc_bare: float
    lulc_snow: float
    lulc_rangeland: float


def metadata_ok(
    dt_hours: ArrayLike,
    lat: ArrayLike,
    month: ArrayLike,
    elev: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Apply the metadata filters: time gap, local winter, optional elevation.

    Args:
        dt_hours: Optical minus radar acquisition time, in hours.
        lat: Latitude in degrees.
        month: Month of the optical acquisition, 1 to 12.
        elev: Elevation in metres. If given, crops above the elevation limit are
            rejected.

    Returns:
        Boolean array, true where the example passes.
    """
    dt_a, lat_a, month_a = (np.asarray(v) for v in (dt_hours, lat, month))
    ok = np.abs(dt_a) <= constants.MAX_DT_HOURS
    winter_north = np.isin(month_a, _WINTER_NORTH) & (lat_a > constants.SNOW_LAT)
    winter_south = np.isin(month_a, _WINTER_SOUTH) & (lat_a < -constants.SNOW_LAT)
    ok &= ~(winter_north | winter_south)
    if elev is not None:
        ok &= np.asarray(elev) <= constants.ELEV_MAX_M
    return ok


def invalid_pixels(
    sar_db: NDArray[np.floating], opt_dn: NDArray[np.number]
) -> bool:
    """Detect NaN or no-data pixels in either modality.

    Args:
        sar_db: Radar in dB.
        opt_dn: Optical digital numbers.

    Returns:
        True if any radar value is not finite, any optical value is not finite, or
        any optical value is zero.
    """
    return bool(
        not np.isfinite(sar_db).all()
        or not np.isfinite(opt_dn).all()
        or (opt_dn == 0).any()
    )


def saturated(opt_rho: NDArray[np.floating]) -> bool:
    """Detect a crop with too many saturated optical pixels.

    Args:
        opt_rho: Optical reflectance.

    Returns:
        True if more than the allowed share of pixels have reflectance of at least
        1.0.
    """
    return bool((opt_rho >= 1.0).mean() > constants.SAT_MAX_FRAC)


def optical_stats(opt_u8: NDArray[np.uint8]) -> OpticalStats:
    """Measure brightness and flatness of the stored 8-bit RGB image.

    Args:
        opt_u8: 8-bit optical image, shape ``[3, H, W]``.

    Returns:
        Mean, standard deviation, near-white share and the analysis-only flat-bright
        flag.
    """
    unit = opt_u8.astype(np.float32) / _MAX_LEVEL
    lum, min_channel = unit.mean(0), unit.min(0)
    return {
        "opt_mean": float(lum.mean()),
        "opt_std": float(lum.std()),
        "opt_white_frac": float((min_channel > _WHITE_LEVEL).mean()),
        "flat_bright": bool(
            lum.std() < _FLAT_STD and min_channel.mean() > _FLAT_MIN_CHANNEL
        ),
    }


def rfi_score(sar_db: NDArray[np.floating]) -> float:
    """Score radio-frequency interference streaks in radar.

    Args:
        sar_db: Radar VV and VH in dB, shape ``[2, H, W]``.

    Returns:
        The largest robust z-score of a row or column mean over both polarisations.
        Only streaks that deviate by more than 3 dB count; 0.0 if there is none.
        Recorded, never a filter.
    """
    worst = 0.0
    for channel in sar_db:
        for axis in (0, 1):
            means = channel.mean(axis=axis)
            median = np.median(means)
            mad = np.median(np.abs(means - median)) * _MAD_TO_SIGMA + _EPS
            if np.abs(means - median).max() > _RFI_MIN_OFFSET_DB:
                worst = max(worst, float((np.abs(means - median) / mad).max()))
    return worst


def flags(sar_db: NDArray[np.floating], dem: NDArray[np.floating]) -> SarDemFlags:
    """Compute radar and terrain scores for one crop.

    Args:
        sar_db: Radar VV and VH in dB, shape ``[2, 128, 128]``.
        dem: Elevation in metres at 20 m pixels, shape ``[128, 128]``.

    Returns:
        Water share, noise-floor share, slope standard deviation and mean elevation.
    """
    grad_y, grad_x = np.gradient(dem.astype(np.float32), _DEM_PIXEL_M)
    slope = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))
    return {
        "water_frac_sar": float((sar_db[0] < constants.WATER_DB).mean()),
        "vh_noise_frac": float((sar_db[1] < constants.NOISE_FLOOR_DB).mean()),
        "slope_std": float(slope.std()),
        "elev_mean": float(dem.mean()),
    }


def steep(slope_std: float) -> bool:
    """Detect terrain too steep for radar to align with optical.

    Args:
        slope_std: Standard deviation of the slope, in degrees.

    Returns:
        True if it exceeds the limit.
    """
    return slope_std > constants.SLOPE_STD_MAX


def spectral_scores(rho12: NDArray[np.floating]) -> SpectralScores:
    """Compute spectral cloud, snow and haze scores from the 12-band crop.

    Args:
        rho12: Reflectance, shape ``[12, H, W]``.

    Returns:
        The cloud-rule share, snow share and haze ratio.
    """
    green, swir = rho12[_B03], rho12[_B11]
    ndsi = (green - swir) / np.maximum(green + swir, _EPS)
    blue, red = rho12[_B02], rho12[_B04]
    snow = (ndsi > _NDSI_SNOW) & (green > _GREEN_SNOW)
    cloud = (
        (blue > _BRIGHT)
        & (red > _BRIGHT)
        & (np.abs(red - blue) < _WHITE_BALANCE)
        & ~snow
    )
    return {
        "cloud_rule_frac": float(cloud.mean()),
        "snow_score": float(snow.mean()),
        "haze_score": float(rho12[_B01].mean() / max(blue.mean(), _EPS)),
    }


def sensei_fracs(mask: NDArray[np.uint8]) -> SenseiFracs:
    """Compute class shares of SSL4EO's neural mask.

    Used for filtering only, never for pairing.

    Args:
        mask: Mask classes on the 128x128 crop.

    Returns:
        Cloud, no-data, water and snow shares.
    """
    return {
        "sensei_cloud_frac": float(np.isin(mask, SENSEI_CLOUD).mean()),
        "sensei_nodata_frac": float((mask == SENSEI_NODATA).mean()),
        "water_frac_opt": float((mask == _SENSEI_WATER).mean()),
        "snow_frac_opt": float((mask == _SENSEI_SNOW).mean()),
    }


def lulc_fracs(lulc: NDArray[np.uint8]) -> LulcFracs:
    """Compute class shares of SSL4EO's land-cover layer.

    A static map, so it is usable for selection.

    Args:
        lulc: Land-cover codes on the 128x128 crop.

    Returns:
        Share of each class (codes in :data:`LULC_CLASSES`).
    """

    def share(code: int) -> float:
        return float((lulc == code).mean())

    return {
        "lulc_water": share(1),
        "lulc_trees": share(2),
        "lulc_flooded": share(3),
        "lulc_crops": share(4),
        "lulc_built": share(5),
        "lulc_bare": share(6),
        "lulc_snow": share(7),
        "lulc_rangeland": share(9),
    }

# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Process one location-season: fetch, score and filter."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from sen_bridge_build import constants, filters, preprocess
from sen_bridge_build.stream import Sample

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

    from sen_bridge_build.stream import Remote

#: Modalities read for every sample.
MODALITIES = ("S2L2A", "S1GRD", "DEM", "LULC")
#: Indices of B04, B03 and B02 in the 12-band L2A stack.
_RGB_BANDS = [3, 2, 1]
#: Mask pixels per 128-pixel crop pixel: the 256-pixel centre crop is subsampled by
#: two.
_MASK_STEP = 2
_FULL_REFLECTANCE = 1.0


class Record(
    filters.OpticalStats,
    filters.SarDemFlags,
    filters.SpectralScores,
    filters.SenseiFracs,
    filters.LulcFracs,
):
    """All scores and flags recorded for one crop.

    Attributes:
        id: Example id, ``<sample id>_<season>``.
        rfi_score: Largest robust z-score of a radar row or column mean. Recorded,
            never a filter.
        sat_frac: Share of optical pixels with reflectance of at least 1.0.
        border_frac: Share of pixels with radar VV below the border-noise level.
            Recorded only.
    """

    id: str
    rfi_score: float
    sat_frac: float
    border_frac: float


class Computed(NamedTuple):
    """The outcome of processing one crop.

    Attributes:
        record: All scores and flags.
        reason: Name of the first filter that rejects the crop, or ``None`` if it is
            kept.
        sar: 8-bit radar image ``[3, 128, 128]``.
        optical: 8-bit optical image ``[3, 128, 128]``.
    """

    record: Record
    reason: str | None
    sar: NDArray[np.uint8]
    optical: NDArray[np.uint8]


def reject_reason(rec: Record, *, invalid: bool) -> str | None:
    """Find the first filter that rejects a crop.

    The order is part of the specification: a crop is reported under the first rule
    it breaks.

    Args:
        rec: The crop's scores and flags.
        invalid: Whether any radar or optical pixel is NaN or no-data.

    Returns:
        The rejection reason, or ``None`` if the crop passes every filter.
    """
    featureless = rec["opt_std"] < constants.WHITE_FLAT_STD
    arid = rec["lulc_bare"] + rec["lulc_rangeland"] >= constants.ARID_LULC_MIN
    water_gap = rec["water_frac_sar"] - rec["water_frac_opt"]
    checks = (
        ("invalid_pixels", invalid or rec["sensei_nodata_frac"] > 0),
        ("saturated", rec["sat_frac"] > constants.SAT_MAX_FRAC),
        ("cloud_shadow", rec["sensei_cloud_frac"] >= constants.CLOUD_MAX_FRAC),
        ("cloud_opaque", rec["cloud_rule_frac"] >= constants.CLOUD_RULE_REJECT),
        (
            "overexposed",
            rec["opt_white_frac"] >= constants.WHITE_MAX_FRAC and featureless,
        ),
        ("too_dark", rec["opt_mean"] < constants.DARK_MEAN_MIN),
        (
            "featureless_arid",
            rec["opt_mean"] > constants.FLAT_BRIGHT_MEAN and featureless and arid,
        ),
        (
            "water_mismatch",
            water_gap > constants.WATER_MISMATCH
            and rec["lulc_water"] > constants.LULC_WATER_MIN,
        ),
        ("steep", filters.steep(rec["slope_std"])),
        ("elevation", rec["elev_mean"] > constants.ELEV_MAX_M),
    )
    return next((name for name, hit in checks if hit), None)


class _Arrays(NamedTuple):
    """The arrays read for one location-season, already cropped to 128x128.

    Attributes:
        sar: Radar VV and VH in dB, ``[2, 128, 128]``.
        digital: B04, B03 and B02 digital numbers at full resolution.
        rho: Optical reflectance (RGB), ``[3, 128, 128]``.
        rho12: Reflectance of all 12 bands, ``[12, 128, 128]``.
        mask: SSL4EO neural mask classes, ``[128, 128]``.
        dem: Elevation in metres, ``[128, 128]``.
        lulc: Land-cover codes, ``[128, 128]``.
    """

    sar: NDArray[np.float32]
    digital: NDArray[np.int16]
    rho: NDArray[np.float32]
    rho12: NDArray[np.float32]
    mask: NDArray[np.uint8]
    dem: NDArray[np.float32]
    lulc: NDArray[np.uint8]


def _load(
    season: int,
    remotes: Mapping[str, Remote],
    members: Mapping[str, tuple[int, int]],
) -> _Arrays:
    """Fetch and crop all arrays of one location-season.

    Args:
        season: Season index 0 to 3.
        remotes: The tar shard of each modality.
        members: ``(offset, size)`` of this sample's zip inside each modality's
            shard.

    Returns:
        The cropped arrays.
    """
    s2, s1, dem_sample, lulc_sample = (
        Sample(remotes[m], *members[m]) for m in ("S2L2A", "S1GRD", "DEM", "LULC")
    )
    bands = s2.s2_all(season)
    lulc = lulc_sample.chunk("bands", f"{season}.0.0.0")[0, 0]
    return _Arrays(
        sar=preprocess.sar_to_db128(s1.s1(season)),
        digital=bands[_RGB_BANDS],
        rho=preprocess.opt_reflectance128(bands[_RGB_BANDS]),
        rho12=preprocess.opt_reflectance128(bands),
        mask=preprocess.center_crop(s2.cloud_mask(season))[
            ::_MASK_STEP, ::_MASK_STEP
        ],
        dem=preprocess.avg_2x2(preprocess.center_crop(dem_sample.dem())),
        lulc=preprocess.center_crop(lulc)[::_MASK_STEP, ::_MASK_STEP],
    )


def compute(
    sample_id: str,
    season: int,
    remotes: Mapping[str, Remote],
    members: Mapping[str, tuple[int, int]],
) -> Computed:
    """Fetch one location-season, score it and apply the filters.

    Args:
        sample_id: Example id, ``<sample id>_<season>``.
        season: Season index 0 to 3.
        remotes: The tar shard of each modality in :data:`MODALITIES`.
        members: ``(offset, size)`` of this sample's zip inside each modality's
            shard.

    Returns:
        The record, the rejection reason (``None`` if kept) and the two 8-bit
        images.
    """
    a = _load(season, remotes, members)
    sar_u8, opt_u8 = preprocess.sar_to_uint8(a.sar), preprocess.opt_to_uint8(a.rho)
    record: Record = {
        "id": sample_id,
        **filters.flags(a.sar, a.dem),
        **filters.spectral_scores(a.rho12),
        **filters.sensei_fracs(a.mask),
        "rfi_score": filters.rfi_score(a.sar),
        "sat_frac": float((a.rho >= _FULL_REFLECTANCE).mean()),
        **filters.optical_stats(opt_u8),
        "border_frac": float((a.sar[0] < constants.BORDER_DB).mean()),
        **filters.lulc_fracs(a.lulc),
    }
    invalid = filters.invalid_pixels(a.sar, a.digital)
    return Computed(record, reject_reason(record, invalid=invalid), sar_u8, opt_u8)

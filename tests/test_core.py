# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests of preprocessing, metadata, splits and the filter rules."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from sen_bridge_build import filters, meta, preprocess, splits
from sen_bridge_build.sample import Record, reject_reason

_SIZE = 264


def test_sar_roundtrip_and_shapes() -> None:
    rng = np.random.default_rng(0)
    db = rng.uniform(-25, -6, (2, _SIZE, _SIZE)).astype(np.float32)
    out = preprocess.sar_to_db128(db)
    assert out.shape == (2, 128, 128)
    u8 = preprocess.sar_to_uint8(out)
    assert u8.shape == (3, 128, 128)
    assert u8.dtype == np.uint8
    assert (
        np.abs(preprocess.sar_from_uint8(u8)[0] - out[0]).max()
        <= 25 / 255 / 2 + 1e-3
    )


def test_multilook_is_linear_power() -> None:
    db = np.full((2, _SIZE, _SIZE), -10.0, np.float32)
    db[:, 4, 4] = 0.0  # the crop starts at 4, so this is block (0, 0)
    expected = 10 * np.log10((1 + 3 * 0.1) / 4)
    assert preprocess.sar_to_db128(db)[0, 0, 0] == pytest.approx(expected, abs=1e-4)


def test_optical_roundtrip() -> None:
    rho = np.linspace(0, 0.3, 3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128)
    back = preprocess.opt_from_uint8(preprocess.opt_to_uint8(rho))
    assert np.abs(back - rho).max() < 0.002
    ends = preprocess.to_model_range(np.array([0, 255], np.uint8))
    assert ends.tolist() == [-1.0, 1.0]


def test_meta_vector() -> None:
    vec = meta.meta_vector(100, 45.0, 10.0, 24.0, orbit_desc=True)
    assert vec.shape == (7,)
    assert vec.dtype == np.float32
    assert vec[5] == pytest.approx(0.5)
    assert vec[6] == 1
    assert np.linalg.norm(vec[2:5]) == pytest.approx(1)


def test_metadata_filter() -> None:
    ok = filters.metadata_ok([10, 60, 10, 10], [50, 0, 50, -50], [7, 7, 1, 1])
    assert ok.tolist() == [True, False, False, True]


def test_region_of_and_other() -> None:
    assert splits.region_of(50, 10) == "Europe"
    assert splits.region_of(-80, 0) == "Other"


def test_splits_and_downsample() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "lat": rng.uniform(-60, 60, 2000),
        "lon": rng.uniform(-180, 180, 2000),
    })
    df["cluster_id"] = splits.assign_clusters(df, df.sample(50, random_state=0))
    df["stratum"] = np.where(df["lat"] > 0, "N", "S")
    df["split"] = splits.enforce_min_distance(df, splits.assign_splits(df))
    assert (df.groupby("cluster_id")["split"].nunique() <= 2).all()
    df["region"] = rng.choice(list("ABC"), len(df), p=[0.6, 0.3, 0.1])
    assert len(splits.stratified_downsample(df, 500)) == 500


def test_cap_share_respects_cap_per_split() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "split": rng.choice(["train", "test"], 1000),
        "w": rng.random(1000),
    })
    out = splits.cap_share(df, df["w"] > 0.7, 0.10)
    for _, group in out.groupby("split"):
        assert (group["w"] > 0.7).mean() <= 0.10 + 1e-9
    assert (out["w"] <= 0.7).sum() == (df["w"] <= 0.7).sum()  # others untouched


def _clean_record() -> Record:
    """Build a record that passes every filter.

    Returns:
        A complete record with unremarkable values.
    """
    return {
        "id": "x_0",
        "opt_mean": 0.5,
        "opt_std": 0.1,
        "opt_white_frac": 0.0,
        "flat_bright": False,
        "water_frac_sar": 0.0,
        "vh_noise_frac": 0.0,
        "slope_std": 2.0,
        "elev_mean": 100.0,
        "cloud_rule_frac": 0.0,
        "snow_score": 0.0,
        "haze_score": 0.8,
        "sensei_cloud_frac": 0.0,
        "sensei_nodata_frac": 0.0,
        "water_frac_opt": 0.0,
        "snow_frac_opt": 0.0,
        "rfi_score": 0.0,
        "sat_frac": 0.0,
        "border_frac": 0.0,
        "lulc_water": 0.0,
        "lulc_trees": 0.2,
        "lulc_flooded": 0.0,
        "lulc_crops": 0.6,
        "lulc_built": 0.2,
        "lulc_bare": 0.0,
        "lulc_snow": 0.0,
        "lulc_rangeland": 0.0,
    }


def test_clean_record_passes() -> None:
    assert reject_reason(_clean_record(), invalid=False) is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"sat_frac": 0.5}, "saturated"),
        ({"sensei_cloud_frac": 0.2}, "cloud_shadow"),
        ({"cloud_rule_frac": 0.9}, "cloud_opaque"),
        ({"opt_white_frac": 0.9, "opt_std": 0.01}, "overexposed"),
        ({"opt_mean": 0.03}, "too_dark"),
        (
            {"opt_mean": 0.95, "opt_std": 0.01, "lulc_bare": 0.9},
            "featureless_arid",
        ),
        ({"water_frac_sar": 0.9, "lulc_water": 0.5}, "water_mismatch"),
        ({"slope_std": 12.0}, "steep"),
        ({"elev_mean": 3000.0}, "elevation"),
    ],
)
def test_each_filter_rejects_with_its_own_reason(
    change: dict[str, float], reason: str
) -> None:
    # The test builds a record from a dict literal; cast it to the TypedDict.
    record = cast("Record", {**_clean_record(), **change})
    assert reject_reason(record, invalid=False) == reason


def test_invalid_pixels_win_over_every_other_reason() -> None:
    # The test builds a record from a dict literal; cast it to the TypedDict.
    record = cast(
        "Record", {**_clean_record(), "sat_frac": 0.5, "elev_mean": 3000.0}
    )
    assert reject_reason(record, invalid=True) == "invalid_pixels"


def test_arid_flat_bright_is_not_rejected_off_arid_land() -> None:
    # The test builds a record from a dict literal; cast it to the TypedDict.
    record = cast("Record", {**_clean_record(), "opt_mean": 0.95, "opt_std": 0.01})
    assert reject_reason(record, invalid=False) is None

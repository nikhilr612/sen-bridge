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
_SHARE = {"A": 0.6, "B": 0.3, "C": 0.1}  # the regional mix that _pool draws from


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


def test_latitude_band_uses_absolute_latitude_and_the_configured_edges() -> None:
    lat = pd.Series([0.0, 19.9, 20.0, -29.9, 30.0, -45.0, 50.0, 80.0])
    assert splits.latitude_band(lat).tolist() == [0, 0, 1, 1, 2, 3, 4, 4]


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
    df["split"] = splits.assign_splits(df, test_rows=300, val_rows=100)
    assert (df.groupby("cluster_id")["split"].nunique() <= 2).all()
    df["region"] = rng.choice(list("ABC"), len(df), p=[0.6, 0.3, 0.1])
    assert len(splits.stratified_downsample(df, 500)) == 500


def test_assign_splits_matches_row_quotas_and_the_stratum_mix() -> None:
    rng = np.random.default_rng(2)
    n = 20_000
    df = pd.DataFrame({"cluster_id": rng.integers(0, 400, n)})
    df["stratum"] = np.where(df["cluster_id"] % 5 == 0, "small", "big")
    split = splits.assign_splits(df, test_rows=3000, val_rows=1000)
    counts = split.value_counts()
    # quotas are met to within about one cluster (50 rows) per stratum
    assert abs(counts["test"] - 3000) <= 150
    assert abs(counts["val"] - 1000) <= 150
    # a whole cluster is never split across two splits
    first = df.assign(s=split).groupby("cluster_id")["s"].transform("first")
    assert (first == split).all()
    # each stratum holds its proportional share of the test pool
    small = (split[df["stratum"] == "small"] == "test").sum() / counts["test"]
    assert small == pytest.approx((df["stratum"] == "small").mean(), abs=0.03)


def test_drop_within_removes_only_rows_near_the_reference() -> None:
    step = 3 / 111.195  # degrees of longitude per 3 km at the equator
    reference = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    rows = pd.DataFrame({"lat": [0.0, 0.0, 0.0], "lon": [step, 2 * step, 1.0]})
    kept = splits.drop_within(rows, reference, km=5.0)
    # 3 km is dropped; 6 km (beyond the rule) and 111 km stay.
    assert kept["lon"].round(4).tolist() == [round(2 * step, 4), 1.0]


def test_drop_within_with_no_reference_returns_everything() -> None:
    rows = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    assert len(splits.drop_within(rows, rows.iloc[0:0])) == 1


def _pool(n: int, seed: int = 0) -> pd.DataFrame:
    """Build a pool with regions and water / bare-ground shares above the caps.

    Args:
        n: Number of rows.
        seed: Seed of the draw.

    Returns:
        Rows with the columns the selection uses.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "region": rng.choice(list("ABC"), n, p=[0.6, 0.3, 0.1]),
        "water_frac_opt": np.where(rng.random(n) < 0.30, 0.95, 0.0),
        "lulc_bare": np.where(rng.random(n) < 0.20, 0.9, 0.0),
    })


def test_stratified_downsample_matches_the_target_mix() -> None:
    pool = _pool(10_000)
    target = {"A": 0.5, "B": 0.4, "C": 0.1}  # differs from the pool's 60/30/10
    chosen = splits.stratified_downsample(pool, 1000, target_share=target, seed=1)
    shares = chosen["region"].value_counts(normalize=True)
    for region, want in target.items():
        assert shares[region] == pytest.approx(want, abs=0.002)
    assert len(chosen) == 1000


def test_stratified_downsample_under_represents_a_short_region() -> None:
    pool = _pool(10_000)
    pool = pool[(pool["region"] != "C") | (pool.index % 50 == 0)]  # C nearly absent
    chosen = splits.stratified_downsample(pool, 1000, target_share=_SHARE)
    assert len(chosen) == 1000
    assert (chosen["region"] == "C").sum() <= (pool["region"] == "C").sum()


def test_select_capped_is_exact_in_size_and_caps() -> None:
    pool = _pool(5000)
    chosen = splits.select_capped(pool, 1000, _SHARE, seed=3)
    assert len(chosen) == 1000
    assert chosen.index.is_unique
    assert splits.is_water(chosen).sum() <= 0.10 * 1000
    assert splits.is_bare(chosen).sum() <= 0.08 * 1000


def test_select_capped_on_a_small_pool_obeys_the_caps_without_padding() -> None:
    pool = _pool(50)
    chosen = splits.select_capped(pool, 1000, _SHARE)
    assert len(chosen) <= 50  # never more rows than the pool holds
    assert splits.is_water(chosen).sum() <= 0.10 * len(chosen)


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

# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Geographic splits, the 5 km leakage check, caps and downsampling."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sen_bridge_build import constants

_EARTH_RADIUS_KM = 6371.0
#: Default shares of clusters assigned to train, validation and test.
_DEFAULT_FRACTIONS = (0.835, 0.033, 0.132)
#: Margin above its pre-filter share that a region may take in the downsample.
_REGION_SLACK = 0.03

#: Coarse region boxes as (name, lat_min, lat_max, lon_min, lon_max). The first
#: match wins.
REGIONS = (
    ("N America", 12, 85, -170, -50),
    ("C/S America", -60, 12, -120, -30),
    ("Europe", 35, 72, -12, 45),
    ("Africa/Middle East", -36, 35, -20, 60),
    ("S/C Asia", 5, 55, 60, 95),
    ("E/SE Asia", -12, 60, 95, 150),
    ("Oceania", -50, -10, 110, 180),
)


def region_of(lat: float, lon: float) -> str:
    """Name the coarse region of a location.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        The first region in :data:`REGIONS` whose box contains the point, else
        ``"Other"``.
    """
    for name, lat_min, lat_max, lon_min, lon_max in REGIONS:
        if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
            return name
    return "Other"


def _unit(lat: pd.Series, lon: pd.Series) -> np.ndarray:
    """Convert coordinates to unit vectors on the sphere.

    Args:
        lat: Latitudes in degrees.
        lon: Longitudes in degrees.

    Returns:
        Array of shape ``[N, 3]``.
    """
    phi, lam = np.deg2rad(lat.to_numpy()), np.deg2rad(lon.to_numpy())
    return np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], -1
    )


def assign_clusters(df: pd.DataFrame, seeds: pd.DataFrame) -> pd.Series:
    """Assign each row to its nearest seed location.

    Args:
        df: Rows with ``lat`` and ``lon`` columns.
        seeds: Seed locations with ``lat`` and ``lon`` columns.

    Returns:
        Series named ``cluster_id`` holding the index of the nearest seed.
    """
    _, nearest = cKDTree(_unit(seeds["lat"], seeds["lon"])).query(
        _unit(df["lat"], df["lon"])
    )
    return pd.Series(nearest, index=df.index, name="cluster_id")


def assign_splits(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = _DEFAULT_FRACTIONS,
    seed: int = 0,
) -> pd.Series:
    """Assign whole clusters to train, validation or test by ``stratum``.

    Args:
        df: Rows with ``cluster_id`` and ``stratum`` columns.
        fractions: Shares of clusters per stratum for train, validation and test;
            only the last two are used, the rest of the clusters go to train.
        seed: Seed of the random assignment.

    Returns:
        Series named ``split`` with values ``"train"``, ``"val"`` or ``"test"``.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=df.index, name="split")
    clusters = df.drop_duplicates("cluster_id")[["cluster_id", "stratum"]]
    for _, group in clusters.groupby("stratum"):
        ids = rng.permutation(group["cluster_id"].to_numpy())
        n_val, n_test = (
            round(len(ids) * fractions[1]),
            round(len(ids) * fractions[2]),
        )
        out[df["cluster_id"].isin(ids[:n_val])] = "val"
        out[df["cluster_id"].isin(ids[n_val : n_val + n_test])] = "test"
    return out


def enforce_min_distance(
    df: pd.DataFrame,
    split: pd.Series,
    km: float = constants.SPLIT_MIN_DIST_KM,
) -> pd.Series:
    """Move validation and test rows that are too close to any train row into train.

    Args:
        df: Rows with ``lat`` and ``lon`` columns.
        split: Split of each row.
        km: Minimum allowed distance to a train row, in kilometres.

    Returns:
        A new split series with the offending rows set to ``"train"``.
    """
    split = split.copy()
    is_train = split == "train"
    if is_train.all() or not is_train.any():
        return split
    tree = cKDTree(_unit(df.loc[is_train, "lat"], df.loc[is_train, "lon"]))
    dist, _ = tree.query(_unit(df.loc[~is_train, "lat"], df.loc[~is_train, "lon"]))
    split[split.index[~is_train][dist * _EARTH_RADIUS_KM < km]] = "train"
    return split


def stratified_downsample(
    df: pd.DataFrame,
    n: int,
    region_col: str = "region",
    pre_share: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Sample ``n`` rows, capping each region at its pre-filter share plus 3 points.

    If the caps cannot reach ``n`` they are relaxed to the available counts.

    Args:
        df: Rows to sample from.
        n: Number of rows wanted. If the frame is not larger, it is returned
            unchanged.
        region_col: Column holding the region of each row.
        pre_share: Pre-filter share of each region. Defaults to the current shares.
        seed: Seed of the random sample.

    Returns:
        The sampled rows in their original order.
    """
    rng = np.random.default_rng(seed)
    if len(df) <= n:
        return df
    share = pre_share or df[region_col].value_counts(normalize=True).to_dict()
    caps = {
        r: int(np.floor((share.get(r, 0) + _REGION_SLACK) * n))
        for r in df[region_col].unique()
    }
    avail = df[region_col].value_counts()
    take = {r: min(int(avail[r]), caps[r]) for r in avail.index}
    while sum(take.values()) < n:  # hand out the shortfall one row at a time
        room = {r: min(int(avail[r]), caps[r]) - take[r] for r in take}
        room = {r: v for r, v in room.items() if v > 0}
        if not room:
            caps = {
                r: int(avail[r]) for r in avail.index
            }  # caps infeasible: relax them
            continue
        take[max(room, key=lambda r: room[r])] += 1
    while sum(take.values()) > n:
        take[max(take, key=lambda r: take[r])] -= 1
    parts = [
        g.iloc[rng.permutation(len(g))[: take[r]]]
        for r, g in df.groupby(region_col)
    ]
    return cast(
        "pd.DataFrame", pd.concat(parts).sort_index()
    )  # pandas-stubs lose the type here


def cap_share(
    df: pd.DataFrame, member: pd.Series, cap: float, seed: int = 0
) -> pd.DataFrame:
    """Cap the share of ``member`` rows in each split at ``cap``.

    Args:
        df: Rows with a ``split`` column.
        member: Boolean series aligned with ``df`` marking the rows to cap.
        cap: Largest allowed share of members in each split.
        seed: Seed of the random choice of members to keep.

    Returns:
        The kept rows in their original order.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in df.groupby("split"):
        flag = member.loc[group.index]
        inside, rest = group[flag], group[~flag]
        max_inside = int(cap * len(rest) / (1 - cap))
        if len(inside) > max_inside:
            inside = inside.iloc[rng.permutation(len(inside))[:max_inside]]
        keep += [rest, inside]
    return cast(
        "pd.DataFrame", pd.concat(keep).sort_index()
    )  # pandas-stubs lose the type here


def cap_water(
    df: pd.DataFrame,
    cap: float = constants.WATER_SPLIT_CAP,
    seed: int = 0,
    col: str = "water_frac_opt",
) -> pd.DataFrame:
    """Cap water-dominated crops per split, judged by the optical mask.

    The optical mask is used because radar VV misses wind-roughened sea.

    Args:
        df: Rows with a ``split`` column and the water-share column.
        cap: Largest allowed share of water-dominated crops in each split.
        seed: Seed of the random choice.
        col: Column holding the water share.

    Returns:
        The kept rows in their original order.
    """
    return cap_share(df, df[col] > constants.WATER_CROP_FRAC, cap, seed)

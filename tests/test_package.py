# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Tests of the release selection: sizes, caps and the 5 km rule across splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from package import select_release
from sen_bridge_build import splits

_KM_PER_DEG = 111.195
_MIN_KM = 5.0


def _world(n: int, seed: int = 0) -> pd.DataFrame:
    """Build rows scattered over a 4x4 degree area, with clusters and a split each.

    Args:
        n: Number of rows.
        seed: Seed of the draw.

    Returns:
        Rows with coordinates, region, caps columns and an assigned split.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "lat": rng.uniform(0, 4, n),
        "lon": rng.uniform(0, 4, n),
        "region": rng.choice(list("AB"), n),
        "water_frac_opt": np.where(rng.random(n) < 0.2, 0.95, 0.0),
        "lulc_bare": np.where(rng.random(n) < 0.15, 0.9, 0.0),
    })
    cell = 4  # quarter-degree cells give enough clusters for every split
    lat_cell = np.floor(df["lat"] * cell).astype(int)
    df["cluster_id"] = lat_cell * 100 + np.floor(df["lon"] * cell).astype(int)
    df["stratum"] = df["region"]
    df["split"] = splits.assign_splits(
        df, test_rows=n // 4, val_rows=n // 8, seed=seed
    )
    return df


def _nearest_km(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Smallest distance between two sets of rows in km, on a flat approximation.

    Args:
        a: Rows with ``lat`` and ``lon``.
        b: Rows with ``lat`` and ``lon``.

    Returns:
        The minimum pairwise distance.
    """
    dist, _ = cKDTree(b[["lat", "lon"]].to_numpy()).query(
        a[["lat", "lon"]].to_numpy()
    )
    return float(np.asarray(dist).min() * _KM_PER_DEG)


def test_selection_has_exact_sizes_and_keeps_every_pair_of_splits_apart() -> None:
    df = _world(6000)
    target = {"test": 300, "val": 100, "train": 1500}
    out, dropped = select_release(df, target, seed=0)
    parts = {k: out[out["split"] == k] for k in target}
    assert {k: len(v) for k, v in parts.items()} == target
    for first, second in (("test", "val"), ("test", "train"), ("val", "train")):
        assert _nearest_km(parts[first], parts[second]) >= _MIN_KM
    assert set(dropped) == {"val_near_test", "train_near_test", "train_near_val"}


def _test_ids(df: pd.DataFrame) -> set[int]:
    """Return the row labels that belong to the test split.

    Args:
        df: Rows with a ``split`` column.

    Returns:
        The labels of the test rows.
    """
    return {int(i) for i in df.index[df["split"] == "test"]}


def test_test_set_is_independent_of_the_later_splits() -> None:
    df = _world(6000)
    small = select_release(df, {"test": 300, "val": 50, "train": 500}, seed=0)[0]
    large = select_release(df, {"test": 300, "val": 200, "train": 2000}, seed=0)[0]
    assert _test_ids(small) == _test_ids(large)

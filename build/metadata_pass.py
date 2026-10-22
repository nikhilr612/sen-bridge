# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Step 1: metadata-only filters on the SSL4EO-S12 v1.1 train metadata.

Usage::

    python build/metadata_pass.py train_metadata.parquet candidates.parquet

Turns the one-row-per-location table into one row per location-season and keeps the
rows that pass the time-gap and local-winter filters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

from sen_bridge_build import filters

_SEASONS = 4


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the location table into one row per location-season.

    Args:
        df: SSL4EO metadata with ``S1_time_k``, ``S2_time_k`` and ``cloud_cover_k``
            columns for ``k`` in 0..3 and ``center_lat`` / ``center_lon`` columns.

    Returns:
        One row per location-season with ``id``, ``season``, ``t_radar``,
        ``t_optical``, ``dt_hours``, ``doy``, ``cloud_cover``, ``lat`` and ``lon``.
    """
    base = df.drop(columns=["geometry"], errors="ignore")
    rows = [
        base.assign(
            season=k,
            t_radar=df[f"S1_time_{k}"],
            t_optical=df[f"S2_time_{k}"],
            cloud_cover=df[f"cloud_cover_{k}"],
        )
        for k in range(_SEASONS)
    ]
    out = pd.concat(rows, ignore_index=True)
    out["dt_hours"] = (out["t_optical"] - out["t_radar"]).dt.total_seconds() / 3600
    out["doy"] = out["t_optical"].dt.dayofyear
    out["id"] = out["sample_id"] + "_" + out["season"].astype(str)
    # pandas-stubs lose the DataFrame return type of ``rename``.
    return cast(
        "pd.DataFrame",
        out.rename(columns={"center_lat": "lat", "center_lon": "lon"}),
    )


def main() -> None:
    """Apply the metadata filters and write the candidate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    df = long_form(pd.read_parquet(args.src))
    keep = filters.metadata_ok(df["dt_hours"], df["lat"], df["t_optical"].dt.month)
    print(
        f"location-seasons: {len(df)} -> {int(keep.sum())} (time gap + snow season)"
    )
    df[keep].to_parquet(args.dst)


if __name__ == "__main__":
    main()

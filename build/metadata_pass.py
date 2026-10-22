"""Step 1: metadata-only filters on SSL4EO-S12 v1.1 train_metadata.parquet.

Usage: python build/metadata_pass.py train_metadata.parquet candidates.parquet
"""
import sys

import pandas as pd

from sen_bridge_build import filters


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    """One row per location-season, with the time gap in hours."""
    rows = []
    for k in range(4):
        d = df.drop(columns=["geometry"], errors="ignore").assign(
            season=k, t_radar=df[f"S1_time_{k}"], t_optical=df[f"S2_time_{k}"],
            cloud_cover=df[f"cloud_cover_{k}"])
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["dt_hours"] = (out.t_optical - out.t_radar).dt.total_seconds() / 3600
    out["doy"] = out.t_optical.dt.dayofyear
    out["id"] = out.sample_id + "_" + out.season.astype(str)
    return out.rename(columns={"center_lat": "lat", "center_lon": "lon"})


def main(src, dst):
    df = long_form(pd.read_parquet(src))
    keep = filters.metadata_ok(df.dt_hours, df.lat, df.t_optical.dt.month)
    print(f"location-seasons: {len(df)} -> {int(keep.sum())} (time gap + snow season)")
    df[keep].to_parquet(dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])

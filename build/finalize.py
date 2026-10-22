"""Step 4: clusters, splits, 5 km leakage check, water cap, meta vector, final table.

Usage: python build/finalize.py PILOT_DIR [--target 150000] [--seed 0]

Deviation from the proposal: the 10,000 seed-city list is not shipped with SSL4EO, so a
cluster is a 2x2 degree cell (a conservative stand-in); the 5 km proximity check still runs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sen_bridge import constants as C, meta, splits

REGIONS = [  # coarse (lat0, lat1, lon0, lon1) boxes, first match wins
    ("N America", 12, 85, -170, -50), ("C/S America", -60, 12, -120, -30),
    ("Europe", 35, 72, -12, 45), ("Africa/Middle East", -36, 35, -20, 60),
    ("S/C Asia", 5, 55, 60, 95), ("E/SE Asia", -12, 60, 95, 150),
    ("Oceania", -50, -10, 110, 180),
]


def region(lat, lon):
    for name, a, b, c, d in REGIONS:
        if a <= lat < b and c <= lon < d:
            return name
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_dir"); ap.add_argument("--target", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = Path(a.pilot_dir)
    df = pd.read_parquet(d / "pilot.parquet").reset_index(drop=True)

    df["region"] = [region(x, y) for x, y in zip(df.lat, df.lon)]
    df["cluster_id"] = (np.floor(df.lat / 2).astype(int) * 1000 + np.floor(df.lon / 2).astype(int))
    df["stratum"] = df.region + np.where(df.lat >= 0, "_N", "_S")
    df["split"] = splits.assign_splits(df, seed=a.seed)
    n0 = (df.split != "train").sum()
    df["split"] = splits.enforce_min_distance(df, df.split)
    print(f"proximity check moved {n0 - (df.split != 'train').sum()} rows to train")

    df = splits.cap_water(df, seed=a.seed)
    df = splits.cap_share(df, df.lulc_bare >= C.BARE_CROP_FRAC, C.BARE_SPLIT_CAP, seed=a.seed)
    out = []
    for name, g in df.groupby("split"):  # downsample within each split separately
        share = g.region.value_counts(normalize=True).to_dict()
        n = int(a.target * len(g) / len(df)) if a.target < len(df) else len(g)
        out.append(splits.stratified_downsample(g, n, pre_share=share, seed=a.seed))
    df = pd.concat(out).reset_index(drop=True)

    df["cloud_frac"] = df.sensei_cloud_frac
    # orbit is not in SSL4EO's metadata: morning local solar time (UTC + lon/15 h) is descending
    t = pd.to_datetime(df.t_radar)
    solar_h = (t.dt.hour + t.dt.minute / 60 + df.lon / 15) % 24
    desc = (solar_h < 12).to_numpy()
    df["orbit"] = np.where(desc, "desc", "asc")
    df["meta"] = list(meta.meta_vector(df.doy, df.lat, df.lon, df.dt_hours, desc))
    df.to_parquet(d / "final.parquet")
    print(df.split.value_counts().to_string())
    print(df.groupby("split").water_frac_opt.apply(lambda s: (s > C.WATER_CROP_FRAC).mean()).round(3).to_string())


if __name__ == "__main__":
    main()

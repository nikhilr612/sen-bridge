"""Pilot / full build worker: stream samples, apply image filters, write PNGs + tables.

Usage: python build/pilot.py candidates.parquet OUT_DIR --shards 20 --workers 64 [--seed 0]

Only candidates with SSL4EO cloud_cover <= 1% are fetched (yield proxy, as in the proposal);
the rule-based cloud test below is what decides inclusion. Nothing but the output is stored.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sen_bridge import constants as C, filters, preprocess as P, stream

COMMIT = "97a91733db962222e1e6b40a836845ddbbb16b86"


def process(row, remotes, index, out: Path):
    s2 = stream.Sample(remotes["S2L2A"][row.tar], *index["S2L2A"][row.tar][row.zarr])
    s1 = stream.Sample(remotes["S1GRD"][row.tar], *index["S1GRD"][row.tar][row.zarr])
    dem_s = stream.Sample(remotes["DEM"][row.tar], *index["DEM"][row.tar][row.zarr])
    bands = s2.s2_all(row.season)
    dn = bands[[3, 2, 1]]  # B04, B03, B02
    sar = P.sar_to_db128(s1.s1(row.season))
    rho = P.opt_reflectance128(dn)
    rho12 = P.opt_reflectance128(bands)
    mask = P._center_crop(s2.cloud_mask(row.season))[::2, ::2]  # same window, nearest-neighbour to 128
    dem = dem_s.dem()
    fl = filters.flags(sar, P._avg2(P._center_crop(dem)))
    rec = {"id": row.id, **fl}
    if filters.invalid_pixels(sar, dn):
        return rec, "invalid_pixels"
    if filters.saturated(rho):
        return rec, "saturated"
    rec.update(filters.spectral_scores(rho12), **filters.sensei_fracs(mask))
    if rec["sensei_cloud_frac"] >= C.CLOUD_MAX_FRAC:
        return rec, "cloud_shadow"
    if rec["cloud_rule_frac"] >= C.CLOUD_RULE_REJECT:
        return rec, "cloud_opaque"
    if rec["sensei_nodata_frac"] > 0:
        return rec, "invalid_pixels"
    if filters.border_noise(sar):
        return rec, "border_noise"
    if filters.rfi_streaks(sar):
        return rec, "rfi"
    if filters.steep(fl["slope_std"]):
        return rec, "steep"
    if fl["elev_mean"] > C.ELEV_MAX_M:
        return rec, "elevation"
    for sub, arr in (("sar", P.sar_to_uint8(sar)), ("optical", P.opt_to_uint8(rho))):
        Image.fromarray(arr.transpose(1, 2, 0)).save(out / sub / f"{row.id}.png", optimize=False)
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates"); ap.add_argument("out")
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    for d in ("sar", "optical"):
        (out / d).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(a.candidates)
    df = df[df.cloud_cover <= C.CLOUD_MAX_FRAC]
    tars = np.random.default_rng(a.seed).permutation(sorted(df.tar.unique()))[:a.shards]
    df = df[df.tar.isin(tars)].reset_index(drop=True)
    print(f"{len(df)} candidates in {len(tars)} shards", flush=True)

    mods = ("S2L2A", "S1GRD", "DEM")
    remotes = {m: {t: stream.Remote(COMMIT, "train", m, t) for t in tars} for m in mods}
    flat = [r for m in mods for r in remotes[m].values()]
    t0 = time.time()
    stream.prefetch_indices(flat, out / "index", workers=a.workers)
    print(f"indexed in {time.time() - t0:.0f}s", flush=True)
    index = {m: {t: stream.tar_index(remotes[m][t], out / "index") for t in tars} for m in mods}
    # members are keyed by zarr file name
    for m in mods:
        for t in tars:
            index[m][t] = {k.split("/")[-1]: v for k, v in index[m][t].items()}

    recs, rejects = [], []
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, r, remotes, index, out): r for r in df.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                rec, why = f.result()
            except Exception as e:  # keep going; record the failure
                rec, why = {"id": r.id}, f"error:{type(e).__name__}"
            (rejects if why else recs).append({**rec, "reason": why} if why else rec)
            if i % 200 == 0:
                print(f"{i}/{len(df)} kept={len(recs)} {i / (time.time() - t0):.1f}/s", flush=True)
    keep = pd.DataFrame(recs).merge(df, on="id")
    keep.drop(columns=["geometry"], errors="ignore").to_parquet(out / "pilot.parquet")
    pd.DataFrame(rejects).to_parquet(out / "rejects.parquet")
    print(pd.DataFrame(rejects).reason.value_counts().to_string() if rejects else "no rejects")
    print(f"kept {len(keep)} of {len(df)}")


if __name__ == "__main__":
    sys.exit(main())

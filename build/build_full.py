"""Full build, step 3: stream every candidate shard, filter, and write one resumable Parquet part per tar shard.

Usage: python build/build_full.py candidates.parquet OUT_DIR [--shard-workers 12] [--sample-workers 16] [--limit-shards N]

A part holds every candidate of the shard: status ('kept' or the reject reason), all scores/flags, and for kept
rows the two PNGs as bytes. Re-running skips finished shards, so an interrupted build resumes. Rejected rows
store no images. Nothing else is written to disk.
"""
import argparse
import io
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from pilot import COMMIT, compute
from sen_bridge_build import constants as C, stream

MODS = ("S2L2A", "S1GRD", "DEM", "LULC")


def png(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr.transpose(1, 2, 0)).save(b, format="PNG")
    return b.getvalue()


def do_shard(tar: str, rows: pd.DataFrame, out: Path, sample_workers: int):
    part = out / "parts" / f"{tar}.parquet"
    if part.exists():
        return tar, "skipped", 0
    remotes = {m: {tar: stream.Remote(COMMIT, "train", m, tar)} for m in MODS}
    with ThreadPoolExecutor(len(MODS)) as ex:  # the four header walks are independent
        idx = dict(zip(MODS, ex.map(lambda m: stream.tar_index(remotes[m][tar], out / "index"), MODS)))
    index = {m: {tar: {k.split("/")[-1]: v for k, v in idx[m].items()}} for m in MODS}

    def one(r):
        try:
            rec, why, sar, opt = compute(r, remotes, index)
            rec["status"] = why or "kept"
            if why is None:
                rec["sar_png"], rec["optical_png"] = png(sar), png(opt)
            return rec
        except Exception as e:  # recorded, not fatal; resume reruns only missing shards
            return {"id": r.id, "status": f"error:{type(e).__name__}"}

    with ThreadPoolExecutor(sample_workers) as ex:
        recs = list(ex.map(one, rows.itertuples()))
    df = pd.DataFrame(recs)
    tmp = part.with_suffix(".tmp")
    part.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp)
    tmp.rename(part)
    return tar, df.status.value_counts().to_dict(), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates"); ap.add_argument("out")
    ap.add_argument("--shard-workers", type=int, default=12)
    ap.add_argument("--sample-workers", type=int, default=16)
    ap.add_argument("--limit-shards", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    df = pd.read_parquet(a.candidates)
    df = df[df.cloud_cover <= C.CLOUD_MAX_FRAC]
    groups = [(t, g) for t, g in df.groupby("tar")]
    if a.limit_shards:
        groups = groups[:a.limit_shards]
    todo = [(t, g) for t, g in groups if not (out / "parts" / f"{t}.parquet").exists()]
    print(f"{len(df)} candidates, {len(groups)} shards, {len(todo)} to do", flush=True)
    t0, done, n = time.time(), 0, 0
    with ThreadPoolExecutor(a.shard_workers) as ex:
        futs = [ex.submit(do_shard, t, g, out, a.sample_workers) for t, g in todo]
        for f in futs:
            try:
                tar, st, k = f.result()
            except Exception:
                traceback.print_exc()
                continue
            done += 1; n += k
            print(f"[{done}/{len(todo)}] {tar} {st}  {n / (time.time() - t0):.1f} samples/s  {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()

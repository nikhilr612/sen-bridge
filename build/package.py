"""Step 4-5: splits, caps, downsampling, and the Hugging Face dataset layout.

Usage: python build/package.py BUILD_DIR candidates.parquet OUT_DIR [--target 150000 --test 20000 --val 5000]

Order of operations (proposal): clusters -> splits -> 5 km leakage check -> water/bare caps -> region-stratified
downsampling *within each split*. Writes data/<split>-xxxxx-of-xxxxx.parquet (images as PNG bytes, typed with the
`huggingface` schema metadata so `load_dataset` decodes them), MANIFEST.sha256 and rejects/rejects.parquet.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import ClassLabel, Features, Image, Sequence, Value

from sen_bridge import constants as C, meta, splits
from finalize import region

SPLITS = {"train": "train", "val": "validation", "test": "test"}
FEATURES = Features({
    "id": Value("string"), "split": Value("string"), "cluster_id": Value("int32"), "region": Value("string"),
    "sar": Image(), "optical": Image(),
    "meta": Sequence(Value("float32"), length=7), "orbit": ClassLabel(names=["asc", "desc"]),
    "lat": Value("float32"), "lon": Value("float32"), "crs": Value("int32"),
    "t_radar": Value("timestamp[s, tz=UTC]"), "t_optical": Value("timestamp[s, tz=UTC]"),
    "dt_hours": Value("float32"), "doy": Value("int32"),
    "slope_std": Value("float32"), "elev_mean": Value("float32"),
    "water_frac_sar": Value("float32"), "water_frac_opt": Value("float32"), "vh_noise_frac": Value("float32"),
    "snow_score": Value("float32"), "haze_score": Value("float32"), "cloud_frac": Value("float32"),
    "rfi_score": Value("float32"), "flat_bright": Value("bool"), "clean": Value("bool"),
    "lulc_bare": Value("float32"), "lulc_rangeland": Value("float32"), "lulc_water": Value("float32"),
})


def load_build(build: Path, cand_path: str):
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(str(build / "parts" / "*.parquet")))]
    allr = pd.concat(parts, ignore_index=True)
    cand = pd.read_parquet(cand_path)[["id", "sample_id", "season", "lat", "lon", "crs", "t_radar", "t_optical", "dt_hours", "doy", "tar"]]
    return allr, cand


def choose(df: pd.DataFrame, target: dict, seed: int) -> pd.DataFrame:
    """Per-split region-stratified downsample to target[split] rows (or all rows if fewer)."""
    out = []
    for name, g in df.groupby("split"):
        n = target[name]
        share = g.region.value_counts(normalize=True).to_dict()
        out.append(splits.stratified_downsample(g, min(n, len(g)), pre_share=share, seed=seed))
    return pd.concat(out)


def clean_flag(d: pd.DataFrame) -> pd.Series:
    water = (d.water_frac_opt > 0.9) & (d.lulc_water > 0.5)
    return (d.sensei_cloud_frac == 0) & ((d.opt_std >= 0.05) | water) & (d.opt_mean > 0.10) & (d.opt_mean < 0.80)


def write_split(df: pd.DataFrame, split_name: str, out: Path, rows_per_shard: int):
    n = max(1, int(np.ceil(len(df) / rows_per_shard)))
    files = []
    for i in range(n):
        d = df.iloc[i * rows_per_shard:(i + 1) * rows_per_shard]
        cols = {
            "sar": [{"bytes": x, "path": None} for x in d.sar_png], "optical": [{"bytes": x, "path": None} for x in d.optical_png],
            "meta": [list(map(float, m)) for m in d.meta], "orbit": (d.orbit == "desc").astype(int).tolist(),
            "t_radar": pd.to_datetime(d.t_radar, utc=True).dt.floor("s"), "t_optical": pd.to_datetime(d.t_optical, utc=True).dt.floor("s"),
        }
        schema = FEATURES.arrow_schema
        table = pa.Table.from_arrays([pa.array(cols[k] if k in cols else d[k], type=schema.field(k).type) for k in schema.names], schema=schema)
        f = out / "data" / f"{SPLITS[split_name]}-{i:05d}-of-{n:05d}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, f, row_group_size=100, compression="zstd")
        files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build"); ap.add_argument("candidates"); ap.add_argument("out")
    ap.add_argument("--target", type=int, default=150_000); ap.add_argument("--test", type=int, default=20_000)
    ap.add_argument("--val", type=int, default=5_000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows-per-shard", type=int, default=7000)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    allr, cand = load_build(Path(a.build), a.candidates)
    status = allr.status.value_counts()
    rejects = allr[allr.status != "kept"][["id", "status"]].rename(columns={"status": "reason"})
    (out / "rejects").mkdir(exist_ok=True); rejects.to_parquet(out / "rejects" / "rejects.parquet")
    df = allr[allr.status == "kept"].merge(cand, on="id")
    df["region"] = [region(x, y) for x, y in zip(df.lat, df.lon)]
    df["cluster_id"] = np.floor(df.lat / 2).astype(int) * 1000 + np.floor(df.lon / 2).astype(int)
    df["stratum"] = df.region + np.where(df.lat >= 0, "_N", "_S")
    df = df.reset_index(drop=True)
    df["split"] = splits.assign_splits(df, seed=a.seed)
    moved_before = (df.split != "train").sum()
    df["split"] = splits.enforce_min_distance(df, df.split)
    print("proximity check moved", moved_before - (df.split != "train").sum(), "rows to train")
    df = splits.cap_water(df, seed=a.seed)
    df = splits.cap_share(df, df.lulc_bare >= C.BARE_CROP_FRAC, C.BARE_SPLIT_CAP, seed=a.seed)
    target = {"train": a.target - a.val - a.test, "val": a.val, "test": a.test}
    df = choose(df, target, a.seed).reset_index(drop=True)

    t = pd.to_datetime(df.t_radar)
    desc = (((t.dt.hour + t.dt.minute / 60 + df.lon / 15) % 24) < 12).to_numpy()
    df["orbit"] = np.where(desc, "desc", "asc")
    df["meta"] = list(meta.meta_vector(df.doy, df.lat, df.lon, df.dt_hours, desc))
    df["cloud_frac"] = df.sensei_cloud_frac
    df["clean"] = clean_flag(df)
    df = df.sample(frac=1, random_state=a.seed).sort_values("split", kind="stable").reset_index(drop=True)

    files = []
    for name in ("train", "val", "test"):
        files += write_split(df[df.split == name], name, out, a.rows_per_shard)
    import shutil  # reference copies; load_dataset never runs them
    for src, dst in (("build", "scripts/build"), ("src/sen_bridge", "scripts/sen_bridge"), ("docs", "docs"), ("tests", "scripts/tests")):
        shutil.copytree(src, out / dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy("pyproject.toml", out / "scripts" / "pyproject.toml")
    (out / "MANIFEST.sha256").write_text("".join(
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}\n" for f in sorted(files + [out / 'rejects' / 'rejects.parquet'])))

    stats = {"candidates_fetched": int(len(allr)), "kept_by_filters": int((allr.status == "kept").sum()),
             "rejects": {k: int(v) for k, v in status.items() if k != "kept"}, "final": df.split.value_counts().to_dict(),
             "clean_share": float(df.clean.mean()), "water_crops": float((df.water_frac_opt > 0.9).mean()),
             "region_share": df.region.value_counts(normalize=True).round(4).to_dict(),
             "split_bytes": {k: int(sum(f.stat().st_size for f in files if f.name.startswith(SPLITS[k]))) for k in SPLITS},
             "dt_le_24h_share": float((df.dt_hours.abs() <= 24).mean())}
    (out / "build_stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()

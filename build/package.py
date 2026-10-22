# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Steps 4-5: splits, caps, downsampling and the Hugging Face dataset layout.

Usage::

    python build/package.py BUILD_DIR candidates.parquet OUT_DIR
        [--target 150000] [--test 20000] [--val 5000]

Order of operations (proposal): clusters, splits, 5 km leakage check, water and bare
caps, then region-stratified downsampling *within each split*. Writes
``data/<split>-xxxxx-of-xxxxx.parquet`` (images as PNG bytes, typed with the
``huggingface`` schema metadata so ``load_dataset`` decodes them),
``MANIFEST.sha256`` and ``rejects/rejects.parquet``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import ClassLabel, Features, Image, Sequence, Value

from sen_bridge_build import constants, meta, splits

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Name of each split in the release, by internal name.
SPLITS = {"train": "train", "val": "validation", "test": "test"}
FEATURES = Features({
    "id": Value("string"),
    "split": Value("string"),
    "cluster_id": Value("int32"),
    "region": Value("string"),
    "sar": Image(),
    "optical": Image(),
    "meta": Sequence(Value("float32"), length=7),
    "orbit": ClassLabel(names=["asc", "desc"]),
    "lat": Value("float32"),
    "lon": Value("float32"),
    "crs": Value("int32"),
    "t_radar": Value("timestamp[s, tz=UTC]"),
    "t_optical": Value("timestamp[s, tz=UTC]"),
    "dt_hours": Value("float32"),
    "doy": Value("int32"),
    "slope_std": Value("float32"),
    "elev_mean": Value("float32"),
    "water_frac_sar": Value("float32"),
    "water_frac_opt": Value("float32"),
    "vh_noise_frac": Value("float32"),
    "snow_score": Value("float32"),
    "haze_score": Value("float32"),
    "cloud_frac": Value("float32"),
    "rfi_score": Value("float32"),
    "flat_bright": Value("bool"),
    "clean": Value("bool"),
    "lulc_bare": Value("float32"),
    "lulc_rangeland": Value("float32"),
    "lulc_water": Value("float32"),
})
_CANDIDATE_COLUMNS = [
    "id",
    "sample_id",
    "season",
    "lat",
    "lon",
    "crs",
    "t_radar",
    "t_optical",
    "dt_hours",
    "doy",
    "tar",
]
#: Order in which the splits are chosen: the benchmark's test set comes first.
_CHOICE_ORDER = ("test", "val", "train")
#: Pool sizes as a multiple of the target: room for the caps and the 5 km drops.
_TEST_POOL_MARGIN = 1.15
_VAL_POOL_MARGIN = 1.7
_CELL_DEG = 1  # clusters are 1x1 degree cells (the seed-city list is not shipped)
_CELL_STRIDE = 1000
_ROW_GROUP = 100
_HOURS_PER_DAY = 24
_DEG_PER_HOUR = 15  # of longitude
_MINUTES_PER_HOUR = 60
_NOON = 12
_REFERENCE_DIRS = (
    ("build", "scripts/build"),
    ("src/sen_bridge", "scripts/sen_bridge"),
    ("src/sen_bridge_build", "scripts/sen_bridge_build"),
    ("docs", "docs"),
    ("tests", "scripts/tests"),
    ("licenses", "licenses"),
)
_REFERENCE_FILES = ("pyproject.toml", "LICENSE", "NOTICE", "README.md")


def load_results(build_dir: Path) -> pd.DataFrame:
    """Read every part file of a build.

    Args:
        build_dir: Build directory containing ``parts/``.

    Returns:
        One row per fetched candidate: status, scores, and PNG bytes for kept crops.
    """
    parts = [
        pd.read_parquet(f) for f in sorted((build_dir / "parts").glob("*.parquet"))
    ]
    return pd.concat(parts, ignore_index=True)


def kept_examples(results: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Join kept rows with their metadata and add region, cluster and stratum.

    Args:
        results: Output of :func:`load_results`.
        candidates: Candidate table with the metadata columns.

    Returns:
        One row per kept crop with ``region``, ``cluster_id`` and ``stratum``
        columns.
    """
    meta_cols = candidates[_CANDIDATE_COLUMNS]
    df = results[results["status"] == "kept"].merge(meta_cols, on="id")
    df["region"] = list(map(splits.region_of, df["lat"], df["lon"]))
    cell_lat = np.floor(df["lat"] / _CELL_DEG).astype(int)
    cell_lon = np.floor(df["lon"] / _CELL_DEG).astype(int)
    df["cluster_id"] = cell_lat * _CELL_STRIDE + cell_lon
    hemisphere = np.where(df["lat"] >= 0, "_N", "_S")
    band = splits.latitude_band(df["lat"]).astype(str)
    df["stratum"] = df["region"] + hemisphere + "_L" + band
    return cast("pd.DataFrame", df.reset_index(drop=True))  # stubs lose the type


def select_release(
    df: pd.DataFrame,
    target: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Choose the three splits in order of importance, keeping them 5 km apart.

    Test is chosen first, then validation, then train. After each split is chosen,
    the rows of the splits still to come that lie within 5 km of it are dropped, so
    the chosen split keeps its exact size and its composition never depends on the
    later choices. The water and bare-ground caps are applied to every split.

    Args:
        df: Rows with ``split`` and ``region`` columns (clusters already assigned).
        target: Wanted number of rows per split.
        seed: Seed of every random choice.

    Returns:
        The chosen rows, and the number of rows dropped for distance, keyed
        ``<later split>_near_<chosen split>``.
    """
    mix = df["stratum"].value_counts(normalize=True)
    share = {str(k): float(v) for k, v in mix.items()}  # the whole dataset's mix
    pools = {str(k): g for k, g in df.groupby("split")}
    chosen: dict[str, pd.DataFrame] = {}
    dropped: dict[str, int] = {}
    for i, name in enumerate(_CHOICE_ORDER):
        pool = pools[name]
        for member_of, cap in splits.CAPS:
            pool = splits.cap_share(pool, member_of(pool), cap, seed=seed)
        chosen[name] = splits.select_capped(
            pool, target[name], share, seed, region_col="stratum"
        )
        for later in _CHOICE_ORDER[i + 1 :]:
            before = len(pools[later])
            pools[later] = splits.drop_within(pools[later], chosen[name])
            dropped[f"{later}_near_{name}"] = before - len(pools[later])
    return pd.concat(chosen.values()), dropped


def clean_flag(d: pd.DataFrame) -> pd.Series:
    """Flag the conservative ``clean`` subset.

    A crop is clean if the neural mask finds no cloud, shadow or no-data, the
    optical is textured (or confirmed open water) and its mean luminance is neither
    dark nor washed out. The flag is for analysis, never for selecting the
    benchmark.

    Args:
        d: Rows with the score columns.

    Returns:
        Boolean series aligned with ``d``.
    """
    water = (d["water_frac_opt"] > constants.CLEAN_WATER_OPT) & (
        d["lulc_water"] > constants.CLEAN_WATER_LULC
    )
    low, high = constants.CLEAN_MEAN_RANGE
    return (
        (d["sensei_cloud_frac"] == 0)
        & ((d["opt_std"] >= constants.CLEAN_MIN_STD) | water)
        & (d["opt_mean"] > low)
        & (d["opt_mean"] < high)
    )


def add_derived(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add orbit, conditioning vector, cloud share and clean flag, then shuffle.

    Orbit is derived from the local solar time of the radar pass (morning is
    descending), because the source has no orbit field.

    Args:
        df: Selected rows.
        seed: Seed of the shuffle.

    Returns:
        The rows with the derived columns, shuffled within each split.
    """
    t = pd.to_datetime(df["t_radar"])
    solar = (
        t.dt.hour + t.dt.minute / _MINUTES_PER_HOUR + df["lon"] / _DEG_PER_HOUR
    ) % (_HOURS_PER_DAY)
    desc = (solar < _NOON).to_numpy()
    df["orbit"] = np.where(desc, "desc", "asc")
    vectors = meta.meta_vector(
        df["doy"], df["lat"], df["lon"], df["dt_hours"], desc
    )
    meta_col = pd.Series(list(vectors), index=df.index, dtype=object)
    # pandas-stubs type ``assign`` too loosely; restore the DataFrame type.
    df = cast("pd.DataFrame", df.assign(meta=meta_col))
    df["cloud_frac"] = df["sensei_cloud_frac"]
    df["clean"] = clean_flag(df)
    shuffled = df.sample(frac=1, random_state=seed)
    ordered = shuffled.sort_values("split", kind="stable").reset_index(drop=True)
    return cast("pd.DataFrame", ordered)  # stubs lose the type


def write_split(
    df: pd.DataFrame,
    split_name: str,
    out: Path,
    rows_per_shard: int,
) -> list[Path]:
    """Write one split as Parquet shards with the Hugging Face schema.

    Args:
        df: Rows of the split.
        split_name: Internal split name.
        out: Release directory.
        rows_per_shard: Rows per shard file.

    Returns:
        The files written.
    """
    n = max(1, int(np.ceil(len(df) / rows_per_shard)))
    schema = FEATURES.arrow_schema
    files = []
    for i in range(n):
        d = df.iloc[i * rows_per_shard : (i + 1) * rows_per_shard]
        special = {
            "sar": [{"bytes": b, "path": None} for b in d["sar_png"]],
            "optical": [{"bytes": b, "path": None} for b in d["optical_png"]],
            "meta": [list(map(float, m)) for m in d["meta"]],
            "orbit": (d["orbit"] == "desc").astype(int).tolist(),
            "t_radar": pd.to_datetime(d["t_radar"], utc=True).dt.floor("s"),
            "t_optical": pd.to_datetime(d["t_optical"], utc=True).dt.floor("s"),
        }
        arrays = [
            pa.array(
                special[k] if k in special else d[k], type=schema.field(k).type
            )
            for k in schema.names
        ]
        table = pa.Table.from_arrays(arrays, schema=schema)
        path = out / "data" / f"{SPLITS[split_name]}-{i:05d}-of-{n:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=_ROW_GROUP, compression="zstd")
        files.append(path)
    return files


def copy_references(out: Path) -> None:
    """Copy the build code, docs and licences into the release for reference.

    ``load_dataset`` never runs these files.

    Args:
        out: Release directory.
    """
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for src, dst in _REFERENCE_DIRS:
        shutil.copytree(src, out / dst, dirs_exist_ok=True, ignore=ignore)
    for name in _REFERENCE_FILES:
        shutil.copy(name, out / "scripts" / name)
    for script in ("evaluate.py", "make_fid_stats.py"):
        shutil.copy(Path("scripts") / script, out / "scripts" / script)


def write_manifest(out: Path, files: Iterable[Path]) -> None:
    """Write SHA-256 checksums of the data files and the rejects table.

    Args:
        out: Release directory.
        files: Data files to list.
    """
    listed = sorted([*files, out / "rejects" / "rejects.parquet"])
    lines = [
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}\n"
        for f in listed
    ]
    (out / "MANIFEST.sha256").write_text("".join(lines), encoding="utf-8")


def build_stats(
    results: pd.DataFrame,
    df: pd.DataFrame,
    files: list[Path],
    dropped: dict[str, int],
) -> dict[str, object]:
    """Summarise the release for the dataset card.

    Args:
        results: Every fetched candidate.
        df: The released rows.
        files: The data files written.
        dropped: Rows dropped for the 5 km rule, by split pair.

    Returns:
        Counts, shares and sizes, JSON-serialisable.
    """
    status = results["status"].value_counts()
    return {
        "candidates_fetched": len(results),
        "kept_by_filters": int((results["status"] == "kept").sum()),
        "rejects": {k: int(v) for k, v in status.items() if k != "kept"},
        "final": df["split"].value_counts().to_dict(),
        "dropped_for_distance": dropped,
        "clean_share": float(df["clean"].mean()),
        "water_crops": float(
            (df["water_frac_opt"] > constants.WATER_CROP_FRAC).mean()
        ),
        "region_share": df["region"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "split_bytes": {
            k: sum(f.stat().st_size for f in files if f.name.startswith(v))
            for k, v in SPLITS.items()
        },
        "dt_le_24h_share": float((df["dt_hours"].abs() <= _HOURS_PER_DAY).mean()),
    }


def main() -> None:
    """Build the release directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--target", type=int, default=170_000)
    parser.add_argument("--test", type=int, default=30_000)
    parser.add_argument("--val", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows-per-shard", type=int, default=7000)
    args = parser.parse_args()
    out: Path = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = load_results(Path(args.build))
    rejects = results[results["status"] != "kept"][["id", "status"]]
    (out / "rejects").mkdir(exist_ok=True)
    rejects.rename(columns={"status": "reason"}).to_parquet(
        out / "rejects" / "rejects.parquet"
    )
    candidates = pd.read_parquet(args.candidates)
    df = kept_examples(results, candidates)
    df["split"] = splits.assign_splits(
        df,
        test_rows=round(args.test * _TEST_POOL_MARGIN),
        val_rows=round(args.val * _VAL_POOL_MARGIN),
        seed=args.seed,
    )
    target = {"train": args.target - args.val - args.test, "val": args.val}
    target["test"] = args.test
    df, dropped = select_release(df, target, args.seed)
    print("dropped for the 5 km rule:", dropped)
    df = add_derived(df.reset_index(drop=True), args.seed)

    files: list[Path] = []
    for name in SPLITS:
        files += write_split(
            df[df["split"] == name], name, out, args.rows_per_shard
        )
    copy_references(out)
    write_manifest(out, files)
    stats = build_stats(results, df, files, dropped)
    (out / "build_stats.json").write_text(
        json.dumps(stats, indent=2, default=int), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()

# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests of a finished release directory.

Usage::

    python build/accept.py RELEASE_DIR [--min-test 20000] [--min-val 3000]
        [--min-train 100000] [--max-fid-floor 50] [--skip-load-dataset]

Each check is independent and reads the release as a user would, not the build's
intermediate files: checksums, split sizes against the plan, the 5 km leakage
guarantee recomputed from the coordinates, every image decoded, the filter rules
re-applied to the released images and scores, the caps, ``load_dataset``, the card
and the FID statistics. Writes ``docs/acceptance_report.md`` into the release and
exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset
from PIL import Image
from scipy.spatial import cKDTree

from sen_bridge.fid_core import frechet_distance, load_stats
from sen_bridge_build import constants, filters, splits
from sen_bridge_build.sample import Record, reject_reason

_IMG = (128, 128, 3)
_BATCH = 500
_EARTH_RADIUS_KM = 6371.0
_SIZE_CAP_GB = 16
_BYTES_PER_GB = 1024**3
_META_LEN = 7
_EPS = 1e-3
#: Largest allowed gap, in percentage points, between a split's regional share and
#: the release's.
_MIX_TOL = 1.0
#: The same for dimensions that are balanced but not stratified (orbit, season).
_LOOSE_TOL = 3.0
#: Day-of-year edges of the four seasons used for the mix check.
_SEASON_EDGES = (0, 80, 172, 264, 366)
_TABLE_COLUMNS = [
    "id",
    "doy",
    "split",
    "cluster_id",
    "region",
    "meta",
    "orbit",
    "lat",
    "lon",
    "dt_hours",
    "slope_std",
    "elev_mean",
    "water_frac_sar",
    "water_frac_opt",
    "cloud_frac",
    "lulc_bare",
    "lulc_rangeland",
    "lulc_water",
    "clean",
]


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping


class Result(NamedTuple):
    """The outcome of one check.

    Attributes:
        name: Short name of the check.
        passed: Whether it passed.
        detail: What was measured, or why it failed.
        seconds: How long the check took (its share, if one step gave several
            results).
    """

    name: str
    passed: bool
    detail: str
    seconds: float = 0.0


def timed(step: Callable[[], Result | list[Result]]) -> list[Result]:
    """Run one check step and record how long it took.

    Args:
        step: A function returning one result or a list of results.

    Returns:
        The results, each carrying its share of the step's duration.
    """
    start = time.perf_counter()
    out = step()
    items = out if isinstance(out, list) else [out]
    share = (time.perf_counter() - start) / len(items)
    return [r._replace(seconds=round(share, 1)) for r in items]


def data_files(release: Path) -> list[Path]:
    """List the data shards of a release.

    Args:
        release: Release directory.

    Returns:
        The Parquet shards under ``data/``, sorted.
    """
    return sorted((release / "data").glob("*.parquet"))


def check_manifest(release: Path) -> Result:
    """Verify every checksum in ``MANIFEST.sha256``.

    Args:
        release: Release directory.

    Returns:
        Passes if every listed file exists and matches, and every data shard is
        listed.
    """
    lines = (release / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    listed = {}
    bad = []
    for line in lines:
        digest, name = line.split("  ", 1)
        listed[name] = digest
        path = release / name
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            bad.append(name)
    unlisted = [
        f.name for f in data_files(release) if f"data/{f.name}" not in listed
    ]
    ok = not bad and not unlisted
    return Result(
        "manifest", ok, f"{len(listed)} files; bad={bad}; unlisted={unlisted}"
    )


def load_table(release: Path) -> pd.DataFrame:
    """Read the non-image columns of every data shard.

    Args:
        release: Release directory.

    Returns:
        One row per released example.
    """
    parts = [
        pq.read_table(f, columns=_TABLE_COLUMNS).to_pandas()
        for f in data_files(release)
    ]
    # pandas-stubs lose the DataFrame type through ``concat``.
    return cast("pd.DataFrame", pd.concat(parts, ignore_index=True))


def check_counts(df: pd.DataFrame, args: argparse.Namespace) -> Result:
    """Check split sizes against the plan.

    Args:
        df: Released rows.
        args: Minimum sizes.

    Returns:
        Passes if every split meets its minimum.
    """
    sizes = df["split"].value_counts().to_dict()
    need = {"train": args.min_train, "val": args.min_val, "test": args.min_test}
    short = {
        k: (sizes.get(k, 0), v) for k, v in need.items() if sizes.get(k, 0) < v
    }
    return Result("split sizes", not short, f"{sizes}; below minimum: {short}")


def check_rows(df: pd.DataFrame) -> list[Result]:
    """Check identifiers and value ranges.

    Args:
        df: Released rows.

    Returns:
        One result each for uniqueness, finiteness and ranges.
    """
    meta = np.stack(df["meta"].to_list())
    numeric = df.drop(columns=["id", "split", "region", "meta", "clean"])
    finite = bool(
        np.isfinite(numeric.to_numpy(dtype=float)).all() and np.isfinite(meta).all()
    )
    in_range = bool(
        (df["dt_hours"].abs() <= constants.MAX_DT_HOURS + _EPS).all()
        and df["lat"].between(-90, 90).all()
        and df["lon"].between(-180, 180).all()
        and df["orbit"].isin([0, 1]).all()
        and meta.shape[1] == _META_LEN
        and (np.abs(meta[:, :5]) <= 1 + _EPS).all()
    )
    return [
        Result(
            "unique ids", df["id"].is_unique, f"{df['id'].nunique()} of {len(df)}"
        ),
        Result("finite values", finite, "no NaN or inf in scores or meta"),
        Result("value ranges", in_range, "|dt|<=48 h, valid lat/lon, orbit, meta"),
    ]


def check_leakage(df: pd.DataFrame) -> Result:
    """Recompute the distance between every pair of splits.

    Args:
        df: Released rows.

    Returns:
        Passes if the nearest pair of rows in any two different splits is at least
        the 5 km guarantee apart.
    """

    def unit(sub: pd.DataFrame) -> np.ndarray:
        phi, lam = (
            np.deg2rad(sub["lat"].to_numpy()),
            np.deg2rad(sub["lon"].to_numpy()),
        )
        return np.stack(
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], -1
        )

    nearest = {}
    for first, second in (("test", "train"), ("val", "train"), ("test", "val")):
        a, b = df[df["split"] == first], df[df["split"] == second]
        dist, _ = cKDTree(unit(b)).query(unit(a))
        nearest[f"{first}-{second}"] = round(
            float(np.asarray(dist).min() * _EARTH_RADIUS_KM), 2
        )
    return Result(
        "5 km leakage guarantee",
        min(nearest.values()) >= constants.SPLIT_MIN_DIST_KM - _EPS,
        f"nearest cross-split distances in km: {nearest}",
    )


def check_mix(df: pd.DataFrame) -> Result:
    """Check each split has the release's mix of region, hemisphere, latitude, etc.

    Region, hemisphere and latitude band are stratified, so their shares must agree
    tightly. Orbit and season are not stratified, only balanced by the finer
    clusters, so they get a looser tolerance.

    Args:
        df: Released rows.

    Returns:
        Passes if no share of any category in any split differs from its share in
        the release by more than that dimension's tolerance, in percentage points.
    """
    season = pd.cut(df["doy"], _SEASON_EDGES, labels=False)
    keys = {
        "region": (df["region"], _MIX_TOL),
        "hemisphere": (
            pd.Series(np.where(df["lat"] >= 0, "N", "S"), index=df.index),
            _MIX_TOL,
        ),
        "latitude band": (splits.latitude_band(df["lat"]), _MIX_TOL),
        "orbit": (df["orbit"], _LOOSE_TOL),
        "season": (season, _LOOSE_TOL),
    }
    failures, worst = [], {}
    for dimension, (values, tol) in keys.items():
        overall = values.value_counts(normalize=True)
        gap_max = 0.0
        for name in df["split"].unique():
            share = values[df["split"] == name].value_counts(normalize=True)
            gap_max = max(
                gap_max,
                float(
                    (
                        (share.reindex(overall.index).fillna(0) - overall).abs()
                        * 100
                    ).max()
                ),
            )
        worst[dimension] = round(gap_max, 2)
        if gap_max > tol:
            failures.append(dimension)
    return Result(
        "mix per split (region, hemisphere, latitude, orbit, season)",
        not failures,
        f"largest gaps from the release mix (pts): {worst}; failing: {failures}",
    )


def check_caps(df: pd.DataFrame) -> Result:
    """Check the water and bare-ground caps per split.

    Args:
        df: Released rows.

    Returns:
        Passes if no split exceeds either cap (plus a one-row tolerance).
    """
    problems, shares = [], {}
    for name, group in df.groupby("split"):
        water = float((group["water_frac_opt"] > constants.WATER_CROP_FRAC).mean())
        bare = float((group["lulc_bare"] >= constants.BARE_CROP_FRAC).mean())
        shares[str(name)] = {"water": round(water, 4), "bare": round(bare, 4)}
        slack = 1 / len(group)
        if water > constants.WATER_SPLIT_CAP + slack:
            problems.append((name, "water", water))
        if bare > constants.BARE_SPLIT_CAP + slack:
            problems.append((name, "bare", bare))
    return Result("caps", not problems, f"shares {shares}; violations {problems}")


def optical_stats_of(png: bytes) -> filters.OpticalStats:
    """Measure a released optical image with the same function the build used.

    Args:
        png: PNG bytes of the optical image.

    Returns:
        Brightness and flatness statistics.

    Raises:
        ValueError: If the image is not 128x128 RGB.
    """
    arr = np.asarray(Image.open(io.BytesIO(png)))
    if arr.shape != _IMG:
        msg = f"unexpected optical shape {arr.shape}"
        raise ValueError(msg)
    return filters.optical_stats(arr.transpose(2, 0, 1))


class ShardScan(NamedTuple):
    """What decoding one data shard found.

    Attributes:
        pairs: Number of image pairs in the shard.
        bad: Number of images that failed to decode or had the wrong shape.
        stats: Optical statistics of each pair, in row order.
    """

    pairs: int
    bad: int
    stats: list[filters.OpticalStats]


def scan_shard(path: Path) -> ShardScan:
    """Decode every image of one shard and measure the optical ones.

    Runs in a worker process, so it is a module-level function.

    Args:
        path: A data shard.

    Returns:
        The number of pairs, the number of bad images and the optical statistics.
    """
    bad, pairs = 0, 0
    stats: list[filters.OpticalStats] = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=_BATCH, columns=["sar", "optical"]
    ):
        sar, optical = (
            batch.column("sar").to_pylist(),
            batch.column("optical").to_pylist(),
        )
        for s, o in zip(sar, optical, strict=True):
            pairs += 1
            try:
                if np.asarray(Image.open(io.BytesIO(s["bytes"]))).shape != _IMG:
                    bad += 1
                stats.append(optical_stats_of(o["bytes"]))
            except (OSError, ValueError):
                bad += 1
                stats.append(
                    filters.OpticalStats(
                        opt_mean=0.5,
                        opt_std=0.5,
                        opt_white_frac=0.0,
                        flat_bright=False,
                    )
                )
    return ShardScan(pairs, bad, stats)


def neutral_record(
    row: Mapping[Hashable, Any], optical: filters.OpticalStats
) -> Record:
    """Build a record for the shipped rules from a released row and its image.

    Fields the release does not carry (saturation, spectral cloud, other land-cover
    classes) are set to values that no rule rejects, so only rules that can be
    checked from the release are tested.

    Args:
        row: A released row (non-image columns).
        optical: Statistics measured on the released optical image.

    Returns:
        A record to pass to :func:`reject_reason`.
    """
    return {
        "id": str(row["id"]),
        **optical,
        "water_frac_sar": float(row["water_frac_sar"]),
        "vh_noise_frac": 0.0,
        "slope_std": float(row["slope_std"]),
        "elev_mean": float(row["elev_mean"]),
        "cloud_rule_frac": 0.0,
        "snow_score": 0.0,
        "haze_score": 0.0,
        "sensei_cloud_frac": float(row["cloud_frac"]),
        "sensei_nodata_frac": 0.0,
        "water_frac_opt": float(row["water_frac_opt"]),
        "snow_frac_opt": 0.0,
        "rfi_score": 0.0,
        "sat_frac": 0.0,
        "border_frac": 0.0,
        "lulc_water": float(row["lulc_water"]),
        "lulc_trees": 0.0,
        "lulc_flooded": 0.0,
        "lulc_crops": 0.0,
        "lulc_built": 0.0,
        "lulc_bare": float(row["lulc_bare"]),
        "lulc_snow": 0.0,
        "lulc_rangeland": float(row["lulc_rangeland"]),
    }


def check_images_and_filters(release: Path, df: pd.DataFrame) -> list[Result]:
    """Decode every image and re-apply the filter rules to the released data.

    The shards are decoded in parallel worker processes. The optical statistics come
    from the released PNGs themselves, so this also checks that the stored images
    agree with the thresholds. Rules that need the raw 12-band reflectance
    (saturation, spectral cloud) cannot be re-checked here.

    Args:
        release: Release directory.
        df: Released rows in shard order.

    Returns:
        Results for decodability and for the re-applied filters.
    """
    files = data_files(release)
    with ProcessPoolExecutor(
        max_workers=min(len(files), os.cpu_count() or 1)
    ) as pool:
        scans = list(pool.map(scan_shard, files))
    stats = [item for scan in scans for item in scan.stats]
    violations: dict[str, int] = {}
    for row, optical in zip(df.to_dict("records"), stats, strict=True):
        reason = reject_reason(neutral_record(row, optical), invalid=False)
        if reason is not None:
            violations[reason] = violations.get(reason, 0) + 1
    pairs, bad = sum(s.pairs for s in scans), sum(s.bad for s in scans)
    return [
        Result("images decode", bad == 0, f"{pairs} pairs, {bad} bad"),
        Result(
            "filters hold on released data",
            not violations,
            f"rows the shipped rules would reject: {violations or 'none'}",
        ),
    ]


def check_load_dataset(release: Path) -> Result:
    """Load every split of the release the way a user would, by streaming.

    Streaming reads the Parquet shards directly and writes no Arrow cache, so it
    checks the card's configuration, the feature types and the image decoding
    without copying the whole dataset to disk. Row counts are checked against the
    shards separately.

    Args:
        release: Release directory.

    Returns:
        Passes if every split yields a correctly typed, decodable example.
    """
    ds = load_dataset(str(release), streaming=True)
    problems = []
    for split in ("train", "validation", "test"):
        row = next(iter(ds[split]))
        features = ds[split].features
        ok = (
            np.asarray(row["sar"]).shape == _IMG
            and np.asarray(row["optical"]).shape == _IMG
            and len(row["meta"]) == _META_LEN
            and features is not None
            and features["orbit"].names == ["asc", "desc"]
        )
        if not ok:
            problems.append(split)
    return Result(
        "load_dataset (streaming)", not problems, f"bad splits: {problems}"
    )


def check_card_and_files(release: Path, df: pd.DataFrame) -> list[Result]:
    """Check the card, the licence files and the size cap.

    Args:
        release: Release directory.
        df: Released rows.

    Returns:
        Results for the card, the required files and the size.
    """
    text = (release / "README.md").read_text(encoding="utf-8")
    header = yaml.safe_load(text.split("---")[1])
    declared = {
        s["name"]: s["num_examples"] for s in header["dataset_info"]["splits"]
    }
    actual = {"train": 0, "validation": 0, "test": 0}
    for name, n in df["split"].value_counts().items():
        actual["validation" if name == "val" else str(name)] = int(n)
    thresholds = json.loads(
        (release / "docs" / "thresholds.json").read_text(encoding="utf-8")
    )
    current = {k: v for k, v in vars(constants).items() if k.isupper()}
    body = json.dumps(current, sort_keys=True, indent=2)
    fresh = hashlib.sha256(body.encode()).hexdigest() == thresholds["sha256"]
    required = [
        "LICENSE",
        "NOTICE",
        "README.md",
        "MANIFEST.sha256",
        "licenses",
        "docs/LICENSES.md",
        "docs/thresholds.json",
        "rejects/rejects.parquet",
        "stats/fid_test.npz",
        "stats/fid_validation.npz",
        "scripts/evaluate.py",
    ]
    missing = [
        p
        for p in required
        if not (release / p).exists() and not (release / "scripts" / p).exists()
    ]
    size_gb = sum(f.stat().st_size for f in data_files(release)) / _BYTES_PER_GB
    return [
        Result(
            "card counts match data",
            declared == actual,
            f"card {declared} vs data {actual}",
        ),
        Result(
            "thresholds snapshot is current",
            fresh and thresholds["sha256"] in text,
            "docs/thresholds.json matches the constants and is cited in the card",
        ),
        Result("required files", not missing, f"missing: {missing}"),
        Result(
            "size cap",
            size_gb < _SIZE_CAP_GB,
            f"{size_gb:.2f} GB of data (cap {_SIZE_CAP_GB})",
        ),
    ]


def check_stats(release: Path, max_floor: float) -> Result:
    """Sanity-check the FID statistics files.

    Args:
        release: Release directory.
        max_floor: Largest acceptable validation-versus-test FID. The floor is
            biased upward at small sample sizes, so this only catches broken files.

    Returns:
        Passes if shapes are right and the validation-versus-test FID is small.
    """
    val = load_stats(release / "stats" / "fid_validation.npz")
    test = load_stats(release / "stats" / "fid_test.npz")
    shapes_ok = all(
        m.shape == (2048,) and s.shape == (2048, 2048) for m, s in (val, test)
    )
    floor = frechet_distance(*val, *test)
    return Result(
        "FID statistics",
        shapes_ok and 0 <= floor < max_floor,
        f"shapes ok={shapes_ok}; validation-vs-test FID floor {floor:.2f}",
    )


def write_report(release: Path, results: list[Result]) -> None:
    """Write the results as Markdown into the release.

    Args:
        release: Release directory.
        results: All check results.
    """
    lines = [
        "# Acceptance report",
        "",
        "| Check | Result | Seconds | Detail |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.seconds} |"
        f" {r.detail} |"
        for r in results
    ]
    out = release / "docs" / "acceptance_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run every check, time each one, and exit non-zero on a failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--min-test", type=int, default=20_000)
    parser.add_argument("--min-val", type=int, default=10_000)
    parser.add_argument("--min-train", type=int, default=100_000)
    parser.add_argument("--max-fid-floor", type=float, default=50.0)
    parser.add_argument("--skip-load-dataset", action="store_true")
    args = parser.parse_args()
    release = Path(args.release)

    df = load_table(release)
    steps = [
        partial(check_manifest, release),
        partial(check_counts, df, args),
        partial(check_rows, df),
        partial(check_leakage, df),
        partial(check_mix, df),
        partial(check_caps, df),
        partial(check_images_and_filters, release, df),
        partial(check_card_and_files, release, df),
        partial(check_stats, release, args.max_fid_floor),
    ]
    if not args.skip_load_dataset:
        steps.append(partial(check_load_dataset, release))
    results = [r for step in steps for r in timed(step)]
    write_report(release, results)
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        print(f"{verdict}  {r.name} [{r.seconds}s]: {r.detail}")
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()

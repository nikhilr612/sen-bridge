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
import sys
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from PIL import Image
from scipy.spatial import cKDTree

from sen_bridge_build import constants, filters
from sen_bridge_build.sample import Record, reject_reason

_IMG = (128, 128, 3)
_BATCH = 500
_EARTH_RADIUS_KM = 6371.0
_SIZE_CAP_GB = 16
_BYTES_PER_GB = 1024**3
_META_LEN = 7
_EPS = 1e-3
_TABLE_COLUMNS = [
    "id",
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


class Result(NamedTuple):
    """The outcome of one check.

    Attributes:
        name: Short name of the check.
        passed: Whether it passed.
        detail: What was measured, or why it failed.
    """

    name: str
    passed: bool
    detail: str


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
    """Recompute the distance from each validation and test crop to the train set.

    Args:
        df: Released rows.

    Returns:
        Passes if the minimum distance is at least the 5 km guarantee.
    """

    def unit(sub: pd.DataFrame) -> np.ndarray:
        phi, lam = (
            np.deg2rad(sub["lat"].to_numpy()),
            np.deg2rad(sub["lon"].to_numpy()),
        )
        return np.stack(
            [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], -1
        )

    train = df[df["split"] == "train"]
    other = df[df["split"] != "train"]
    dist, _ = cKDTree(unit(train)).query(unit(other))
    nearest = float(np.asarray(dist).min() * _EARTH_RADIUS_KM)
    return Result(
        "5 km leakage guarantee",
        nearest >= constants.SPLIT_MIN_DIST_KM - _EPS,
        f"nearest val/test-to-train distance {nearest:.2f} km",
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


def check_images_and_filters(release: Path, df: pd.DataFrame) -> list[Result]:
    """Decode every image and re-apply the filter rules to the released data.

    The optical statistics come from the released PNGs themselves, so this also
    checks that the stored images agree with the thresholds. Rules that need the raw
    12-band reflectance (saturation, spectral cloud) cannot be re-checked here.

    Args:
        release: Release directory.
        df: Released rows in shard order.

    Returns:
        Results for decodability and for the re-applied filters.
    """
    rows, bad_images, offset = [], 0, 0
    for path in data_files(release):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=_BATCH, columns=["sar", "optical"]):
            sar, optical = (
                batch.column("sar").to_pylist(),
                batch.column("optical").to_pylist(),
            )
            for s, o in zip(sar, optical, strict=True):
                try:
                    if np.asarray(Image.open(io.BytesIO(s["bytes"]))).shape != _IMG:
                        bad_images += 1
                    stats = optical_stats_of(o["bytes"])
                except (OSError, ValueError):
                    bad_images += 1
                    stats = filters.OpticalStats(
                        opt_mean=0.5,
                        opt_std=0.5,
                        opt_white_frac=0.0,
                        flat_bright=False,
                    )
                rows.append(stats)
            offset += len(sar)
    stats_df = pd.DataFrame(rows)
    violations: dict[str, int] = {}
    for i in range(len(df)):
        r = df.iloc[i]
        rec: Record = {
            "id": str(r["id"]),
            "opt_mean": float(stats_df["opt_mean"].iloc[i]),
            "opt_std": float(stats_df["opt_std"].iloc[i]),
            "opt_white_frac": float(stats_df["opt_white_frac"].iloc[i]),
            "flat_bright": bool(stats_df["flat_bright"].iloc[i]),
            "water_frac_sar": float(r["water_frac_sar"]),
            "vh_noise_frac": 0.0,
            "slope_std": float(r["slope_std"]),
            "elev_mean": float(r["elev_mean"]),
            "cloud_rule_frac": 0.0,
            "snow_score": 0.0,
            "haze_score": 0.0,
            "sensei_cloud_frac": float(r["cloud_frac"]),
            "sensei_nodata_frac": 0.0,
            "water_frac_opt": float(r["water_frac_opt"]),
            "snow_frac_opt": 0.0,
            "rfi_score": 0.0,
            "sat_frac": 0.0,
            "border_frac": 0.0,
            "lulc_water": float(r["lulc_water"]),
            "lulc_trees": 0.0,
            "lulc_flooded": 0.0,
            "lulc_crops": 0.0,
            "lulc_built": 0.0,
            "lulc_bare": float(r["lulc_bare"]),
            "lulc_snow": 0.0,
            "lulc_rangeland": float(r["lulc_rangeland"]),
        }
        reason = reject_reason(rec, invalid=False)
        if reason is not None:
            violations[reason] = violations.get(reason, 0) + 1
    return [
        Result(
            "images decode", bad_images == 0, f"{offset} pairs, {bad_images} bad"
        ),
        Result(
            "filters hold on released data",
            not violations,
            f"rows the shipped rules would reject: {violations or 'none'}",
        ),
    ]


def check_load_dataset(release: Path, df: pd.DataFrame) -> Result:
    """Load the release the way a user would.

    Args:
        release: Release directory.
        df: Released rows (for the expected counts).

    Returns:
        Passes if counts, feature types and a decoded example are right.
    """
    from datasets import load_dataset  # ruff: ignore[import-outside-top-level]

    ds = load_dataset(str(release))
    expected = {"train": "train", "validation": "val", "test": "test"}
    counts_ok = all(
        len(ds[k]) == int((df["split"] == v).sum()) for k, v in expected.items()
    )
    row = ds["test"][0]
    sample_ok = (
        np.asarray(row["sar"]).shape == _IMG
        and np.asarray(row["optical"]).shape == _IMG
        and len(row["meta"]) == _META_LEN
        and ds["test"].features["orbit"].names == ["asc", "desc"]
    )
    return Result(
        "load_dataset",
        counts_ok and sample_ok,
        f"{ {k: len(v) for k, v in ds.items()} }",
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
    from sen_bridge.fid_core import frechet_distance, load_stats  # ruff: ignore[import-outside-top-level]

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
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {r.detail} |"
        for r in results
    ]
    out = release / "docs" / "acceptance_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run every check and exit non-zero if one fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--min-test", type=int, default=20_000)
    parser.add_argument("--min-val", type=int, default=3_000)
    parser.add_argument("--min-train", type=int, default=100_000)
    parser.add_argument("--max-fid-floor", type=float, default=50.0)
    parser.add_argument("--skip-load-dataset", action="store_true")
    args = parser.parse_args()
    release = Path(args.release)

    df = load_table(release)
    results = [check_manifest(release), check_counts(df, args), *check_rows(df)]
    results += [check_leakage(df), check_caps(df)]
    results += check_images_and_filters(release, df)
    results += check_card_and_files(release, df)
    results.append(check_stats(release, args.max_fid_floor))
    if not args.skip_load_dataset:
        results.append(check_load_dataset(release, df))
    write_report(release, results)
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()

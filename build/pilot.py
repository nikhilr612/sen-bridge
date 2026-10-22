# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Pilot build: stream a few shards, filter, and write PNG pairs plus tables for auditing.

Usage::

    python build/pilot.py candidates.parquet OUT_DIR [--shards 20] [--workers 64] [--seed 0]

Unlike ``build_full.py``, this writes images of kept crops to ``OUT_DIR/{sar,optical}`` and up to
150 examples of each rejection reason to ``OUT_DIR/rejects/<reason>/``, which the audit tools read.
It also writes ``pilot.parquet`` (kept rows) and ``rejects.parquet``.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from PIL import Image

from build_full import open_shard
from sen_bridge_build import constants
from sen_bridge_build.sample import MODALITIES, compute

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sen_bridge_build.stream import Remote

    Opened = tuple[dict[str, Remote], dict[str, dict[str, tuple[int, int]]]]

#: Saved example images per rejection reason, for the audit.
REJECT_CAP = 150
_PROGRESS_EVERY = 200


def save_pair(
    out: Path, sub: str, sample_id: str, sar: NDArray[np.uint8], opt: NDArray[np.uint8]
) -> None:
    """Write the radar and optical PNGs of one crop.

    Args:
        out: Pilot directory.
        sub: Sub-directory (empty for kept crops, ``rejects/<reason>`` otherwise).
        sample_id: Example id used as the file name.
        sar: 8-bit radar image ``[3, 128, 128]``.
        opt: 8-bit optical image ``[3, 128, 128]``.
    """
    for name, arr in (("sar", sar), ("optical", opt)):
        directory = out / sub / name
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.transpose(1, 2, 0)).save(directory / f"{sample_id}.png")


def select(path: Path, shards: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """Choose the candidates of a random subset of shards.

    Args:
        path: Candidate table.
        shards: Number of shards to use.
        seed: Seed of the shard choice.

    Returns:
        The candidates of the chosen shards and the shard names.
    """
    candidates = pd.read_parquet(path)
    candidates = candidates[candidates["cloud_cover"] <= constants.CLOUD_MAX_FRAC]
    tars = list(np.random.default_rng(seed).permutation(sorted(candidates["tar"].unique())))
    chosen = [str(t) for t in tars[:shards]]
    subset = cast("pd.DataFrame", candidates[candidates["tar"].isin(chosen)])  # stubs lose the type
    return cast("pd.DataFrame", subset.reset_index(drop=True)), chosen


def run_pilot(
    candidates: pd.DataFrame,
    opened: dict[str, Opened],
    out: Path,
    workers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Process all candidates, saving images, and split the results into kept and rejected.

    Args:
        candidates: Candidates with ``id``, ``season``, ``zarr`` and ``tar`` columns.
        opened: Opened shards by name.
        out: Pilot directory.
        workers: Number of samples processed concurrently.

    Returns:
        Records of kept crops and records (with a ``reason``) of rejected or failed ones.
    """
    saved: Counter[str] = Counter()
    lock = threading.Lock()

    def process(row: tuple[str, int, str, str]) -> tuple[dict[str, object], str | None]:
        sample_id, season, zarr, tar = row
        remotes, index = opened[tar]
        done = compute(sample_id, season, remotes, {m: index[m][zarr] for m in MODALITIES})
        if done.reason is None:
            save_pair(out, "", sample_id, done.sar, done.optical)
        else:
            with lock:
                saved[done.reason] += 1
                keep = saved[done.reason] <= REJECT_CAP
            if keep:
                save_pair(out, f"rejects/{done.reason}", sample_id, done.sar, done.optical)
        return dict(done.record), done.reason

    rows = list(
        zip(
            candidates["id"],
            candidates["season"].astype(int),
            candidates["zarr"],
            candidates["tar"],
            strict=True,
        )
    )
    kept: list[dict[str, object]] = []
    rejects: list[dict[str, object]] = []
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(process, r): r for r in rows}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                record, reason = future.result()
            except Exception as exc:  # ruff: ignore[blind-except]  (keep going; the failure is recorded)
                record, reason = {"id": futures[future][0]}, f"error:{type(exc).__name__}"
            if reason is None:
                kept.append(record)
            else:
                rejects.append({**record, "reason": reason})
            if i % _PROGRESS_EVERY == 0:
                rate = i / (time.time() - start)
                print(f"{i}/{len(rows)} kept={len(kept)} {rate:.1f}/s", flush=True)
    return kept, rejects


def main() -> None:
    """Run the pilot build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)

    candidates, tars = select(Path(args.candidates), args.shards, args.seed)
    print(f"{len(candidates)} candidates in {len(tars)} shards", flush=True)
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        opened = dict(
            zip(tars, pool.map(lambda t: open_shard(t, out / "index"), tars), strict=True)
        )
    print(f"indexed in {time.time() - start:.0f}s", flush=True)

    kept, rejects = run_pilot(candidates, opened, out, args.workers)
    extra = candidates.drop(columns=["geometry"], errors="ignore")
    merged = pd.DataFrame(kept).merge(extra, on="id")
    merged.to_parquet(out / "pilot.parquet")
    rejected = pd.DataFrame(rejects)
    rejected.to_parquet(out / "rejects.parquet")
    print(rejected["reason"].value_counts().to_string() if len(rejected) else "no rejects")
    print(f"kept {len(merged)} of {len(candidates)}")


if __name__ == "__main__":
    main()

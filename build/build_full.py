# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Full build, step 3: stream shards, filter, write one Parquet part per shard.

Usage::

    python build/build_full.py candidates.parquet OUT_DIR
        [--shard-workers 12] [--sample-workers 16] [--limit-shards N]
        [--partition I/N]

A part holds every candidate of the shard: status (``kept`` or the reject reason),
all scores and flags, and for kept rows the two PNGs as bytes. ``--partition I/N``
handles every N-th shard starting at I, so N processes can share the work.
Re-running skips finished shards, so an interrupted build resumes. Rejected rows
store no images. Nothing else is written to disk.
"""

from __future__ import annotations

import argparse
import io
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PIL import Image

from sen_bridge_build import constants, stream
from sen_bridge_build.sample import MODALITIES, compute

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

#: Hugging Face commit of SSL4EO-S12 v1.1 that the build reads (recorded in the
#: datasheet).
COMMIT = "97a91733db962222e1e6b40a836845ddbbb16b86"


def to_png(arr: NDArray[np.uint8]) -> bytes:
    """Encode a channel-first 8-bit image as PNG.

    Args:
        arr: Image of shape ``[3, H, W]``.

    Returns:
        The PNG file contents.
    """
    buffer = io.BytesIO()
    Image.fromarray(arr.transpose(1, 2, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def process_row(
    sample_id: str,
    season: int,
    zarr: str,
    remotes: Mapping[str, stream.Remote],
    index: Mapping[str, Mapping[str, tuple[int, int]]],
) -> dict[str, object]:
    """Process one candidate; a failure becomes a status, not an exception.

    Args:
        sample_id: Example id, ``<sample id>_<season>``.
        season: Season index 0 to 3.
        zarr: Name of the sample's zip inside the shards.
        remotes: The shard of each modality.
        index: Tar index of each modality's shard.

    Returns:
        A row for the part file: id, status, scores, and PNG bytes if the crop is
        kept.
    """
    try:
        members = {m: index[m][zarr] for m in MODALITIES}
        done = compute(sample_id, season, remotes, members)
    # Recorded, not fatal: a rerun retries only the missing shards.
    except Exception as exc:  # ruff: ignore[blind-except]
        return {"id": sample_id, "status": f"error:{type(exc).__name__}"}
    row: dict[str, object] = {**done.record, "status": done.reason or "kept"}
    if done.reason is None:
        row["sar_png"], row["optical_png"] = to_png(done.sar), to_png(done.optical)
    return row


def open_shard(
    tar: str,
    cache_dir: Path,
) -> tuple[dict[str, stream.Remote], dict[str, dict[str, tuple[int, int]]]]:
    """Open one shard of every modality and index its zip members.

    Args:
        tar: Shard name.
        cache_dir: Directory of cached tar indices.

    Returns:
        The remote of each modality and, for each, a map from zip name to ``(offset,
        size)``.
    """
    remotes = {m: stream.Remote(COMMIT, f"train/{m}/{tar}") for m in MODALITIES}
    with ThreadPoolExecutor(
        len(MODALITIES)
    ) as pool:  # the four header walks are independent
        indices = list(
            pool.map(lambda m: stream.tar_index(remotes[m], cache_dir), MODALITIES)
        )
    index = {
        m: {name.split("/")[-1]: v for name, v in idx.items()}
        for m, idx in zip(MODALITIES, indices, strict=True)
    }
    return remotes, index


def do_shard(
    tar: str,
    rows: pd.DataFrame,
    out: Path,
    sample_workers: int,
) -> tuple[str, object, int]:
    """Process every candidate of one shard and write its part file.

    Args:
        tar: Shard name.
        rows: Candidates of the shard with ``id``, ``season`` and ``zarr`` columns.
        out: Build directory.
        sample_workers: Number of samples processed concurrently.

    Returns:
        The shard name, a summary (status counts, or ``"skipped"``) and the number
        of rows.
    """
    part = out / "parts" / f"{tar}.parquet"
    if part.exists():
        return tar, "skipped", 0
    remotes, index = open_shard(tar, out / "index")
    jobs = list(zip(rows["id"], rows["season"], rows["zarr"], strict=True))
    with ThreadPoolExecutor(sample_workers) as pool:
        results = list(
            pool.map(
                lambda j: process_row(
                    str(j[0]), int(j[1]), str(j[2]), remotes, index
                ),
                jobs,
            )
        )
    frame = pd.DataFrame(results)
    tmp = part.with_suffix(".tmp")
    part.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(tmp)
    tmp.rename(part)
    return tar, frame["status"].value_counts().to_dict(), len(frame)


def main() -> None:
    """Run the full build over all shards of the candidate table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--shard-workers", type=int, default=12)
    parser.add_argument("--sample-workers", type=int, default=16)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument(
        "--partition", default="0/1", help="I/N: every N-th shard from I"
    )
    args = parser.parse_args()
    candidates = pd.read_parquet(args.candidates)
    candidates = candidates[candidates["cloud_cover"] <= constants.CLOUD_MAX_FRAC]
    groups = list(candidates.groupby("tar"))
    if args.limit_shards:
        groups = groups[: args.limit_shards]
    index, count = (int(x) for x in args.partition.split("/"))
    groups = groups[index::count]
    todo = [
        (t, g)
        for t, g in groups
        if not (args.out / "parts" / f"{t}.parquet").exists()
    ]
    print(
        f"{len(candidates)} candidates, {len(groups)} shards, {len(todo)} to do",
        flush=True,
    )
    start, done, total = time.time(), 0, 0
    with ThreadPoolExecutor(args.shard_workers) as pool:
        futures = [
            pool.submit(do_shard, str(t), g, args.out, args.sample_workers)
            for t, g in todo
        ]
        for future in futures:
            try:
                tar, summary, count = future.result()
            # Log and continue; a rerun resumes the missing shards.
            except Exception:  # ruff: ignore[blind-except]
                traceback.print_exc()
                continue
            done += 1
            total += count
            elapsed = time.time() - start
            print(
                f"[{done}/{len(todo)}] {tar} {summary}  "
                f"{total / elapsed:.1f} samples/s  {elapsed / 60:.0f} min",
                flush=True,
            )


if __name__ == "__main__":
    main()

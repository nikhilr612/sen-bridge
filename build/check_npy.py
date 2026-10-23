# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Check the ``.npy`` release against the Parquet release it was made from.

Usage::

    python build/check_npy.py SOURCE_DIR NPY_DIR

Every row of every split is re-read from the Parquet shards, decoded, and compared
with the arrays on disk (pixels, ids, ``meta``), then the manifest, the metadata
table, the shapes and the loader are checked. Exits non-zero on any failure.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from export_npy import sha256_file
from sen_bridge import data

_BATCH = 1000
_SIDE = 128


def check_split(source: Path, out: Path, split: str) -> list[str]:
    """Compare one split's arrays with its Parquet shards, row by row.

    Args:
        source: Parquet release directory.
        out: ``.npy`` release directory.
        split: Split name.

    Returns:
        Descriptions of the problems found (empty when clean).
    """
    problems: list[str] = []
    opened = data.Split(out, split)
    row = 0
    for shard in sorted((source / "data").glob(f"{split}-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=_BATCH):
            n = batch.num_rows
            table = batch.to_pydict()
            for key in ("sar", "optical"):
                got = getattr(opened, key)[row : row + n]
                for i, cell in enumerate(table[key]):
                    png = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
                    if not np.array_equal(got[i], np.asarray(png)):
                        problems.append(f"{split} row {row + i}: {key} differs")
            if opened.ids[row : row + n].tolist() != table["id"]:
                problems.append(f"{split} rows {row}-{row + n}: ids differ")
            expect = np.asarray(table["meta"], np.float32)
            if not np.array_equal(opened.meta[row : row + n], expect):
                problems.append(f"{split} rows {row}-{row + n}: meta differs")
            row += n
    if row != len(opened):
        problems.append(f"{split}: {row} source rows but {len(opened)} on disk")
    return problems[:20]


def main() -> None:
    """Run all checks and exit non-zero if any fails."""
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    failures: list[str] = []

    digests = data.parse_manifest(out / data.MANIFEST)
    files = [p for p in out.rglob("*") if p.is_file() and p.name != data.MANIFEST]
    failures += [
        f"manifest: {p.name} differs or is unlisted"
        for p in files
        if digests.get(str(p.relative_to(out))) != sha256_file(p)
    ]
    sys.stdout.write(f"manifest: {len(digests)} entries checked\n")

    meta = pd.read_parquet(out / data.METADATA_FILE)
    all_ids: list[str] = []
    for split in data.SPLITS:
        opened = data.Split(out, split)
        for key in ("sar", "optical"):
            array = getattr(opened, key)
            if array.dtype != np.uint8 or array.shape[1:] != (_SIDE, _SIDE, 3):
                failures.append(f"{split}: bad {key} {array.dtype} {array.shape}")
        rows = meta[meta["split"] == split]
        if rows["row"].tolist() != list(range(len(opened))):
            failures.append(f"{split}: metadata rows not aligned")
        if rows["id"].tolist() != opened.ids.tolist():
            failures.append(f"{split}: metadata ids differ from ids.npy")
        all_ids += opened.ids.tolist()
        problems = check_split(source, out, split)
        failures += problems
        sys.stdout.write(
            f"{split}: {len(opened):,} rows, {len(problems)} problems\n"
        )
    if len(set(all_ids)) != len(all_ids):
        failures.append("ids are not unique across splits")
    for name in ("README.md", "LICENSE", "NOTICE"):
        if not (out / name).exists():
            failures.append(f"missing {name}")

    sys.stdout.write(
        "\n".join(failures) + "\n" if failures else "ALL CHECKS PASS\n"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

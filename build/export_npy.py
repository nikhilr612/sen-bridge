# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Export the Parquet release as memory-mappable ``.npy`` files.

Usage::

    python build/export_npy.py SOURCE_DIR OUT_DIR

``SOURCE_DIR`` holds the Parquet release (``data/*.parquet``, ``LICENSE``,
``NOTICE``). ``OUT_DIR`` receives, per split, ``<split>_{sar,optical,meta,ids}.npy``
with rows in the Parquet order, plus ``metadata.parquet`` (every other column and a
``row`` index), the licence files, the card and ``MANIFEST.sha256``.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from card import CITATION
from sen_bridge import data

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Release split names, as in the Parquet file names.
SPLITS = data.SPLITS
_SIZE = 128
_BATCH = 512
_IMAGE_COLUMNS = ("sar", "optical")
_CHUNK = 1 << 20
_REPO = "nikhilr01/sen-bridge"

_CARD = """---
license: cc-by-4.0
pretty_name: SEN-Bridge (npy)
task_categories:
- image-to-image
tags:
- remote-sensing
- sentinel-1
- sentinel-2
- sar
- flow-matching
viewer: false
---

# SEN-Bridge, memory-mappable `.npy` release

The same 170,000 radar-to-optical pairs as
[`nikhilr01/sen-bridge`](https://huggingface.co/datasets/nikhilr01/sen-bridge)
(@@TRAIN@@ train / @@VAL@@ validation / @@TEST@@ test; see that card for how
the data were built, filtered and split), stored as raw 8-bit arrays so that random
access is a slice of a memory-mapped file with no image decoding. It is an
optional extra copy; the Parquet release remains the reference. The pixels are
identical to the decoded PNGs of that release (checked on every row by comparing
arrays, see below).

## Files

| File | Contents |
|---|---|
| `<split>_sar.npy` | `uint8 [N, 128, 128, 3]`, R=VV, G=VH, B=VV-VH |
| `<split>_optical.npy` | `uint8 [N, 128, 128, 3]`, RGB (B4, B3, B2) |
| `<split>_meta.npy` | `float32 [N, 7]` conditioning vectors |
| `<split>_ids.npy` | `[N]` example ids |
| `metadata.parquet` | the other columns of the main release, plus `row` |
| `MANIFEST.sha256` | SHA-256 of every file |

`<split>` is `train`, `validation` or `test`. Row `i` of the arrays of a split is
row `i` of its Parquet shards (files in name order) and the row with `row == i` and
that `split` in `metadata.parquet`, whose `split` column uses these names (the
main release writes `val` there for validation). `meta` is `(sin doy, cos doy, x,
y, z, dt / 48 h, orbit)`, the orbit being +1 (descending) or -1 (ascending).

## Loading

With the [`sen-bridge`](https://github.com/nikhilr612/sen-bridge) package, which
downloads from a pinned revision and checks every file:

```python
from sen_bridge import data

train = data.load_dataset("train")          # or "validation", "test"
pair = train[1234]                          # sar, optical, meta, id
batch = train[[5, 1, 9]]                    # arrays with a leading batch axis
for batch in train.iter_batches(256, shuffle=True, seed=0):
    ...
```

A `Split` has `__len__` and `__getitem__`, so it works as a map-style dataset in
PyTorch and Grain, and it pickles by path so worker processes reopen the memory
maps.
`load_dataset("train", float_images=True)` returns `float32` images in `[-1, 1]`.
Without the package:

```python
import numpy as np

sar = np.load("train_sar.npy", mmap_mode="r")   # read after `hf download`
```

Size: about @@GB@@ GB of arrays. Reads are fastest from a local SSD with the files
in the page cache.

## License and attribution

CC BY 4.0 (`LICENSE`), the licence of the source dataset SSL4EO-S12 v1.1. Contains
modified Copernicus Sentinel data 2019-2021. Attribution, changes made and the
third-party notices are in `NOTICE` and in the main dataset card.
"""


def sha256_file(path: Path) -> str:
    """Hash a file in chunks.

    Args:
        path: File to hash.

    Returns:
        The hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def decode(png: bytes) -> NDArray[np.uint8]:
    """Decode one PNG to an RGB array.

    Args:
        png: Encoded image.

    Returns:
        The ``[128, 128, 3]`` array.

    Raises:
        ValueError: If the image has an unexpected shape.
    """
    arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
    if arr.shape != (_SIZE, _SIZE, 3):
        msg = f"unexpected image shape {arr.shape}"
        raise ValueError(msg)
    return arr


def export_split(files: list[Path], split: str, out: Path) -> pd.DataFrame:
    """Write the four arrays of one split.

    Args:
        files: Parquet shards of the split, in order.
        split: Split name.
        out: Output directory.

    Returns:
        The non-image columns of the split, with a ``row`` column.
    """
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    images = {
        key: np.lib.format.open_memmap(
            out / data.array_name(split, key),
            mode="w+",
            dtype=np.uint8,
            shape=(total, _SIZE, _SIZE, 3),
        )
        for key in _IMAGE_COLUMNS
    }
    ids: list[str] = []
    metas: list[NDArray[np.float32]] = []
    tables: list[pd.DataFrame] = []
    row = 0
    for shard in files:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=_BATCH):
            n = batch.num_rows
            for key in _IMAGE_COLUMNS:
                column = batch.column(key).to_pylist()
                for i, cell in enumerate(column):
                    images[key][row + i] = decode(cell["bytes"])
            ids.extend(batch.column("id").to_pylist())
            metas.append(np.asarray(batch.column("meta").to_pylist(), np.float32))
            rest = batch.drop_columns([*_IMAGE_COLUMNS, "meta"]).to_pandas()
            rest.insert(0, "row", np.arange(row, row + n, dtype=np.int32))
            # The Parquet release labels validation rows ``val``; use the name of
            # the split's files so the table can be filtered by it.
            rest["split"] = split
            tables.append(rest)
            row += n
    for array in images.values():
        array.flush()
    np.save(out / data.array_name(split, "id"), np.array(ids))
    np.save(out / data.array_name(split, "meta"), np.concatenate(metas))
    return pd.concat(tables, ignore_index=True)


def write_manifest(out: Path) -> None:
    """Write ``MANIFEST.sha256`` for every other file under ``out``.

    Args:
        out: Release directory.
    """
    lines = [
        f"{sha256_file(p)}  {p.relative_to(out)}"
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != data.MANIFEST and not p.name.endswith(".ok")
    ]
    (out / data.MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_card(out: Path, counts: dict[str, int]) -> None:
    """Write the dataset card.

    Args:
        out: Release directory.
        counts: Examples per split.
    """
    gigabytes = sum(p.stat().st_size for p in out.glob("*.npy")) / 1e9
    text = (
        _CARD
        .replace("@@TRAIN@@", f"{counts['train']:,}")
        .replace("@@VAL@@", f"{counts['validation']:,}")
        .replace("@@TEST@@", f"{counts['test']:,}")
        .replace("@@GB@@", f"{gigabytes:.1f}")
    )
    (out / "README.md").write_text(text + CITATION, encoding="utf-8")


def main() -> None:
    """Export all splits, then write the table, licences, card and manifest."""
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for split in SPLITS:
        files = sorted((source / "data").glob(f"{split}-*.parquet"))
        sys.stdout.write(f"{split}: {len(files)} shards\n")
        table = export_split(files, split, out)
        counts[split] = len(table)
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_parquet(out / data.METADATA_FILE)
    shutil.copy(source / "LICENSE", out / "LICENSE")
    shutil.copy(source / "NOTICE", out / "NOTICE")
    write_card(out, counts)
    write_manifest(out)
    sys.stdout.write(f"wrote {sum(counts.values()):,} pairs to {out}\n")


if __name__ == "__main__":
    main()

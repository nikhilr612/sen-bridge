# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Tests of the ``.npy`` export and the memory-mapped loader."""

from __future__ import annotations

import io

# The test pickles only its own, trusted object.
import pickle  # ruff: ignore[suspicious-pickle-import]
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

import export_npy
from sen_bridge import data

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_N = 7
_SIDE = 128


def _png(array: NDArray[np.uint8]) -> dict[str, object]:
    """Encode an image as the Parquet release stores it.

    Args:
        array: ``[128, 128, 3]`` image.

    Returns:
        The ``{"bytes", "path"}`` cell of an image column.
    """
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def _images(seed: int) -> NDArray[np.uint8]:
    """Draw random images.

    Args:
        seed: Seed of the draw.

    Returns:
        ``[N, 128, 128, 3]`` pixels.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (_N, _SIDE, _SIDE, 3), dtype=np.uint8)


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """Build a small release by exporting two Parquet shards per split.

    Args:
        tmp_path: Temporary directory.

    Returns:
        The exported release directory.
    """
    source = tmp_path / "src" / "data"
    source.mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    for k, split in enumerate(data.SPLITS):
        sar, opt = _images(k), _images(10 + k)
        for shard, rows in enumerate((slice(0, 4), slice(4, _N))):
            ids = [f"{split}-{i}" for i in range(_N)][rows]
            table = pa.table({
                "id": ids,
                "split": ["val" if split == "validation" else split] * len(ids),
                "sar": [_png(a) for a in sar[rows]],
                "optical": [_png(a) for a in opt[rows]],
                "meta": [[float(i)] * 7 for i in range(_N)][rows],
                "lat": [float(i) for i in range(_N)][rows],
            })
            pq.write_table(table, source / f"{split}-{shard:05d}-of-00002.parquet")
    tables = [
        export_npy.export_split(
            sorted(source.glob(f"{split}-*.parquet")), split, out
        )
        for split in data.SPLITS
    ]
    pd.concat(tables).to_parquet(out / data.METADATA_FILE)
    export_npy.write_manifest(out)
    return out


def test_rows_are_aligned_and_pixels_exact(release: Path) -> None:
    train = data.Split(release, "train")
    assert len(train) == _N
    assert np.array_equal(train.sar, _images(0))
    assert np.array_equal(train.optical, _images(10))
    assert train.ids.tolist() == [f"train-{i}" for i in range(_N)]
    assert np.array_equal(train.meta[:, 0], np.arange(_N, dtype=np.float32))
    meta = train.metadata()
    assert meta["row"].tolist() == list(range(_N))
    assert meta["lat"].tolist() == [float(i) for i in range(_N)]


def test_indexing_modes(release: Path) -> None:
    split = data.Split(release, "validation")
    assert len(split.metadata()) == _N  # source rows say "val", not "validation"
    one = split[3]
    assert one["sar"].shape == (_SIDE, _SIDE, 3)
    assert one["meta"].shape == (7,)
    assert str(one["id"]) == "validation-3"
    batch = split[[5, 1, 5]]
    assert batch["sar"].shape == (3, _SIDE, _SIDE, 3)
    assert batch["id"].tolist() == ["validation-5", "validation-1", "validation-5"]
    assert split[1:3]["sar"].shape[0] == 2
    assert split[np.array([0, 6])]["meta"].shape == (2, 7)


def test_float_images_are_scaled_to_unit_range(release: Path) -> None:
    split = data.Split(release, "test", float_images=True)
    image = split[0]["sar"]
    raw = data.Split(release, "test")[0]["sar"]
    assert image.dtype == np.float32
    assert np.allclose(image, raw.astype(np.float32) / 127.5 - 1)
    assert image.min() >= -1
    assert image.max() <= 1


def test_pickle_reopens_instead_of_copying(release: Path) -> None:
    split = data.Split(release, "train", float_images=True)
    blob = pickle.dumps(split)
    assert len(blob) < 10_000
    # The bytes were produced two lines above.
    clone = pickle.loads(blob)  # ruff: ignore[suspicious-pickle-usage]
    assert clone.float_images
    assert np.array_equal(clone[2]["sar"], split[2]["sar"])


def test_iter_batches_covers_each_example_once(release: Path) -> None:
    split = data.Split(release, "train")
    seen = [
        i
        for batch in split.iter_batches(3, shuffle=True, seed=1)
        for i in batch["id"].tolist()
    ]
    assert sorted(seen) == sorted(split.ids.tolist())
    assert seen != split.ids.tolist()
    sizes = [len(b["id"]) for b in split.iter_batches(3, drop_last=True)]
    assert sizes == [3, 3]
    assert len(list(split.iter_batches(4, epochs=2))) == 4


def test_mismatched_lengths_are_rejected(release: Path) -> None:
    np.save(
        release / data.array_name("train", "meta"), np.zeros((2, 7), np.float32)
    )
    with pytest.raises(ValueError, match="different lengths"):
        data.Split(release, "train")


def test_download_verifies_every_file(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = {p.name: p for p in release.iterdir() if p.is_file()}
    fetched: list[str] = []

    def fake(url: str, dst: Path) -> None:
        name = url.rsplit("/", 1)[1]
        fetched.append(name)
        dst.write_bytes(served[name].read_bytes())

    manifest = release / data.MANIFEST
    monkeypatch.setattr(data.urllib.request, "urlretrieve", fake)
    monkeypatch.setattr(data, "MANIFEST_SHA256", data._sha256(manifest))
    target = tmp_path / "dl"
    data.download(target, ["train"])
    assert fetched[0] == data.MANIFEST
    assert len(fetched) == 5
    split = data.load_dataset("train", root=target)
    assert isinstance(split, data.Split)
    assert np.array_equal(split.sar, _images(0))
    assert len(fetched) == 5  # nothing is fetched twice

    served[data.array_name("test", "meta")].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="does not match"):
        data.download(target, ["test"])
    assert not (target / data.array_name("test", "meta")).exists()

# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests of range-request streaming against a synthetic tar of zarr zips."""

from __future__ import annotations

import io
import tarfile
import zipfile
from typing import TYPE_CHECKING

import numpy as np
from numcodecs import Blosc
from typing_extensions import override

from sen_bridge_build import stream

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_SHAPE = (4, 2, 8, 8)


class FakeRemote(stream.Remote):
    """An in-memory stand-in for a remote shard that counts requests."""

    def __init__(self, data: bytes, name: str = "train/S1GRD/x.tar") -> None:
        """Wrap raw tar bytes (no network involved).

        Args:
            data: The tar file contents.
            name: Name reported as the remote path.
        """
        self.data, self.name, self.size, self.calls = data, name, len(data), 0

    @override
    def get(self, start: int, end: int) -> bytes:
        """Serve a byte range from memory and count the request.

        Args:
            start: First byte.
            end: One past the last byte.

        Returns:
            The requested bytes.
        """
        self.calls += 1
        return self.data[start : min(end, self.size)]


def _tar_with_zarr(arr: NDArray[np.float16]) -> bytes:
    """Build a tar holding two identical zarr zips, the first behind a pax header.

    Args:
        arr: Array stored in both zips.

    Returns:
        The tar file contents.
    """
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("bands/.zarray", '{"chunks": [4, 2, 8, 8], "dtype": "<f2"}')
        z.writestr("bands/0.0.0.0", Blosc().encode(arr.astype("<f2").tobytes()))
    payload = zipped.getvalue()
    tarred = io.BytesIO()
    with tarfile.open(fileobj=tarred, mode="w", format=tarfile.PAX_FORMAT) as t:
        first = tarfile.TarInfo("a.zarr.zip")
        first.size = len(payload)
        first.pax_headers = {"k": "v"}
        t.addfile(first, io.BytesIO(payload))
        second = tarfile.TarInfo("b.zarr.zip")
        second.size = len(payload)
        t.addfile(second, io.BytesIO(payload))
    return tarred.getvalue()


def test_index_and_chunk_roundtrip(tmp_path: Path) -> None:
    arr = np.random.default_rng(0).normal(-10, 3, _SHAPE).astype("<f2")
    remote = FakeRemote(_tar_with_zarr(arr))
    index = stream.tar_index(remote, tmp_path)
    assert set(index) == {"a.zarr.zip", "b.zarr.zip"}
    for name, (offset, size) in index.items():
        got = stream.Sample(remote, offset, size).s1(2)
        assert np.allclose(got, arr[2].astype(np.float32)), name
    assert stream.tar_index(FakeRemote(b""), tmp_path) == index  # served from cache


def test_zip_directory_is_read_with_few_requests() -> None:
    arr = np.zeros(_SHAPE, "<f2")
    remote = FakeRemote(_tar_with_zarr(arr))
    offset, size = stream.tar_index(remote)["b.zarr.zip"]
    before = remote.calls
    stream.Sample(remote, offset, size).s1(0)
    assert remote.calls - before <= 3

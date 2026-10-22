# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Stream SSL4EO-S12 v1.1 samples straight from Hugging Face with HTTP range requests.

Nothing is written to disk except small cached tar indices. A shard is a tar of per-sample Zarr-v2
zips; a season of S2L2A is a single blosc chunk (about 1 MB), so only the seasons that survive the
metadata filters are fetched.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import requests
from huggingface_hub import hf_hub_url
from numcodecs import Blosc
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

#: Hugging Face dataset repository of SSL4EO-S12 v1.1.
REPO = "embed2scale/SSL4EO-S12-v1.1"
_BLOCK = 512  # tar block size
#: Pax header, its data block and the real header: read in one request.
_PAX_SKIP = 3 * _BLOCK
_ATTEMPTS = 8
_TIMEOUT_S = 60
_MAX_BACKOFF_S = 60
_RETRY_STATUS = frozenset({
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
})
_EXPIRED_STATUS = frozenset({HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND})
_READ_AHEAD = 65536
_SEEK_SET, _SEEK_CUR, _SEEK_END = 0, 1, 2
#: Indices of B04, B03 and B02 in the 12-band L2A stack.
_RGB_BANDS = [3, 2, 1]
#: Zarr key of an array stored as one chunk (not a network address).
_WHOLE_ARRAY = "0.0.0.0"  # ruff: ignore[hardcoded-bind-all-interfaces]


class Remote:
    """A remote file read with HTTP range requests over one resolved CDN URL.

    The ``HfFileSystem`` costs about 2 s per call, the resolved CDN URL about 0.7 s, and the latter
    parallelises well.

    Attributes:
        name: Path of the file inside the repository.
        size: Size of the file in bytes.
    """

    def __init__(
        self,
        commit: str,
        split: str,
        modality: str,
        shard: str,
        session: requests.Session | None = None,
    ) -> None:
        """Resolve the CDN URL of one shard.

        Args:
            commit: Hugging Face commit hash that pins the dataset revision.
            split: Dataset split, for example ``"train"``.
            modality: Modality folder, for example ``"S2L2A"``.
            shard: Shard file name.
            session: HTTP session to reuse; a new one is created if omitted.
        """
        self.name = f"{split}/{modality}/{shard}"
        self._hub_url = hf_hub_url(REPO, self.name, repo_type="dataset", revision=commit)
        self._session = session or requests.Session()
        self._url = ""
        self.size = 0
        self._resolve()

    def _resolve(self) -> None:
        """Resolve the signed CDN URL and file size."""
        response = self._session.head(self._hub_url, allow_redirects=True, timeout=_TIMEOUT_S)
        response.raise_for_status()
        self._url, self.size = response.url, int(response.headers["content-length"])

    def get(self, start: int, end: int) -> bytes:
        """Read a byte range, retrying with backoff on throttling and network errors.

        Args:
            start: First byte.
            end: One past the last byte; clipped to the file size.

        Returns:
            The bytes in ``[start, end)``.

        Raises:
            OSError: If the range cannot be read after all attempts.
        """
        end = min(end, self.size)
        for attempt in range(_ATTEMPTS):
            try:
                response = self._session.get(
                    self._url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=_TIMEOUT_S
                )
            except requests.RequestException:
                time.sleep(min(2**attempt, _MAX_BACKOFF_S))
                continue
            if response.status_code in _EXPIRED_STATUS:  # signed URL expired
                self._resolve()
                continue
            if response.status_code in _RETRY_STATUS:
                time.sleep(min(2**attempt, _MAX_BACKOFF_S))
                continue
            response.raise_for_status()
            return cast("bytes", response.content)
        msg = f"range read failed for {self.name}"
        raise OSError(msg)


def tar_index(remote: Remote, cache_dir: Path | None = None) -> dict[str, tuple[int, int]]:
    """Index a tar shard by walking its headers with ranged reads.

    Args:
        remote: The shard.
        cache_dir: Directory of cached indices. If given, a cached index is reused and a new one is
            written.

    Returns:
        Map from member name to ``(data offset, size)``.
    """
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / (remote.name.replace("/", "__") + ".json")
        if cache.exists():
            return {k: (v[0], v[1]) for k, v in json.loads(cache.read_text()).items()}
    out: dict[str, tuple[int, int]] = {}
    offset = 0
    while offset + _BLOCK <= remote.size:
        block = remote.get(offset, offset + _PAX_SKIP)
        if block[:_BLOCK] == b"\0" * _BLOCK:
            break  # end of archive
        info = tarfile.TarInfo.frombuf(block[:_BLOCK], "utf-8", "strict")
        header = _BLOCK
        if info.type in {tarfile.XHDTYPE, tarfile.XGLTYPE}:
            header += _BLOCK * ((info.size + _BLOCK - 1) // _BLOCK) + _BLOCK
            info = tarfile.TarInfo.frombuf(block[header - _BLOCK : header], "utf-8", "strict")
        out[info.name] = (offset + header, info.size)
        offset += header + _BLOCK * ((info.size + _BLOCK - 1) // _BLOCK)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


class RangeView(io.RawIOBase):
    """Read-only window ``[offset, offset + size)`` of a :class:`Remote` for ``zipfile`` to seek in.

    One read-ahead block is kept so that the many small reads of a zip directory cost one request.
    """

    def __init__(self, remote: Remote, offset: int, size: int) -> None:
        """Create the window.

        Args:
            remote: The file to read.
            offset: First byte of the window inside the file.
            size: Length of the window.
        """
        super().__init__()
        self._remote, self._base, self._size, self._pos = remote, offset, size, 0
        self._block_start, self._block = 0, b""

    @override
    def readable(self) -> bool:
        """Report that the view can be read.

        Returns:
            Always True.
        """
        return True

    @override
    def seekable(self) -> bool:
        """Report that the view supports seeking.

        Returns:
            Always True.
        """
        return True

    @override
    def tell(self) -> int:
        """Return the current position.

        Returns:
            Offset from the start of the window.
        """
        return self._pos

    @override
    def seek(self, offset: int, whence: int = _SEEK_SET) -> int:
        """Move the position.

        Args:
            offset: Offset relative to ``whence``.
            whence: 0 for the start, 1 for the current position, 2 for the end.

        Returns:
            The new position.
        """
        base = {_SEEK_SET: 0, _SEEK_CUR: self._pos, _SEEK_END: self._size}[whence]
        self._pos = base + offset
        return self._pos

    @override
    def read(self, size: int = -1) -> bytes:
        """Read bytes from the window, fetching a read-ahead block when needed.

        Args:
            size: Number of bytes to read, or -1 for the rest of the window.

        Returns:
            Up to ``size`` bytes; empty at the end of the window.
        """
        want = self._size - self._pos if size < 0 else min(size, self._size - self._pos)
        if want <= 0:
            return b""
        if not (
            self._block_start <= self._pos
            and self._pos + want <= self._block_start + len(self._block)
        ):
            if (
                self._pos + max(want, _READ_AHEAD) >= self._size
            ):  # near the end: fetch the tail block
                start, end = max(0, min(self._pos, self._size - _READ_AHEAD)), self._size
            else:
                start, end = self._pos, self._pos + max(want, _READ_AHEAD)
            self._block = self._remote.get(self._base + start, self._base + end)
            self._block_start = start
        lo = self._pos - self._block_start
        data = self._block[lo : lo + want]
        self._pos += len(data)
        return data

    @override
    def readinto(self, buffer: Any) -> int:
        """Read into a writable buffer.

        Args:
            buffer: Destination buffer.

        Returns:
            Number of bytes read.
        """
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class Sample:
    """One sample's Zarr zip, read chunk by chunk.

    Array metadata (``.zarray``) is identical across samples of a modality, so it is cached on the
    class.
    """

    _meta: ClassVar[dict[tuple[str, str], dict[str, Any]]] = {}

    def __init__(self, remote: Remote, offset: int, size: int) -> None:
        """Open the zip inside a tar shard.

        Args:
            remote: The tar shard.
            offset: Offset of the zip's data inside the shard.
            size: Size of the zip in bytes.
        """
        self._modality = remote.name.split("/")[1]
        self._zip = zipfile.ZipFile(RangeView(remote, offset, size))

    def attrs(self, name: str) -> dict[str, Any]:
        """Read the Zarr array metadata, using the shared cache.

        Args:
            name: Array name, for example ``"bands"``.

        Returns:
            The parsed ``.zarray`` document.
        """
        key = (self._modality, name)
        if key not in self._meta:
            self._meta[key] = json.loads(self._zip.read(f"{name}/.zarray"))
        return self._meta[key]

    def chunk(self, array: str, index: str) -> NDArray[Any]:
        """Decode one chunk.

        Args:
            array: Array name, for example ``"bands"``.
            index: Chunk index, for example ``"1.0.0.0"``.

        Returns:
            The decoded chunk with the array's chunk shape.
        """
        meta = self.attrs(array)
        raw = Blosc().decode(self._zip.read(f"{array}/{index}"))
        return np.frombuffer(cast("bytes", raw), dtype=meta["dtype"]).reshape(meta["chunks"])

    def s2_all(self, season: int) -> NDArray[np.int16]:
        """Read all 12 L2A bands of one season.

        Args:
            season: Season index 0 to 3.

        Returns:
            Digital numbers of shape ``[12, 264, 264]`` in the order B01, B02, B03, B04, B05, B06,
            B07, B08, B8A, B09, B11, B12.
        """
        return self.chunk("bands", f"{season}.0.0.0")[0]

    def cloud_mask(self, season: int) -> NDArray[np.uint8]:
        """Read SSL4EO's own neural (SEnSeI v2) mask.

        Args:
            season: Season index 0 to 3.

        Returns:
            Classes of shape ``[264, 264]``: 0 land, 1 water, 2 snow, 3 thin cloud, 4 thick cloud,
            5 cloud shadow, 6 no data.
        """
        return self.chunk("cloud_mask", f"{season}.0.0")[0]

    def s2_rgb(self, season: int) -> NDArray[np.int16]:
        """Read the red, green and blue bands (B04, B03, B02) of one season.

        Args:
            season: Season index 0 to 3.

        Returns:
            Digital numbers of shape ``[3, 264, 264]``.
        """
        return self.s2_all(season)[_RGB_BANDS]

    def s1(self, season: int) -> NDArray[np.float32]:
        """Read radar VV and VH of one season.

        Args:
            season: Season index 0 to 3.

        Returns:
            Backscatter in dB of shape ``[2, 264, 264]``.
        """
        return self.chunk("bands", _WHOLE_ARRAY)[season].astype(np.float32)

    def dem(self) -> NDArray[np.float32]:
        """Read the elevation model.

        Returns:
            Elevation in metres of shape ``[264, 264]``.
        """
        return self.chunk("bands", _WHOLE_ARRAY)[0, 0].astype(np.float32)


def prefetch_indices(
    remotes: Iterable[Remote],
    cache_dir: Path | None,
    workers: int = 32,
) -> None:
    """Index many shards at once.

    Each shard's header walk is serial, so the work is parallelised across shards.

    Args:
        remotes: Shards to index.
        cache_dir: Directory of cached indices.
        workers: Number of concurrent shards.
    """

    def index(remote: Remote) -> dict[str, tuple[int, int]]:
        return tar_index(remote, cache_dir)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(index, remotes))

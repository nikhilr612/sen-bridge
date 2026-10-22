# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests of range-request streaming against a synthetic tar of zarr zips."""

from __future__ import annotations

import io
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import requests
from numcodecs import Blosc
from typing_extensions import override

from sen_bridge_build import stream

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_SHAPE = (4, 2, 8, 8)
# A dummy value that only checks the header format; it is not a real credential.
_DUMMY_TOKEN = "hf_dummy"  # ruff: ignore[hardcoded-password-string]


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


class _RecordingSession(requests.Session):
    """A session that answers the hub request itself and records its headers."""

    def __init__(self) -> None:
        """Start with no recorded request."""
        super().__init__()
        self.sent_headers: dict[str, str] = {}

    @override
    def head(self, url: str | bytes, **kwargs: Any) -> requests.Response:
        """Record the request headers and answer like the hub.

        Args:
            url: Requested URL.
            **kwargs: Request options; ``headers`` is recorded.

        Returns:
            A successful response with a CDN URL and a size.
        """
        self.sent_headers = dict(kwargs.get("headers") or {})
        response = requests.Response()
        response.status_code = 200
        response.url = "https://cdn.example/shard"
        response.headers["content-length"] = "123"
        return response


def test_token_is_sent_to_the_hub() -> None:
    session = _RecordingSession()
    remote = stream.Remote(
        "c", "train/S2L2A/x.tar", session=session, token=_DUMMY_TOKEN
    )
    assert session.sent_headers == {"Authorization": f"Bearer {_DUMMY_TOKEN}"}
    assert remote.size == 123


def test_no_token_means_no_authorization_header() -> None:
    session = _RecordingSession()
    stream.Remote("c", "train/S2L2A/x.tar", session=session, token="")
    assert session.sent_headers == {}


class _FlakySession(_RecordingSession):
    """A session whose first requests fail like a DNS outage, then succeed."""

    def __init__(self, failures: int) -> None:
        """Set how many requests fail before one succeeds.

        Args:
            failures: Number of initial failing requests.
        """
        super().__init__()
        self.failures, self.calls = failures, 0

    @override
    def head(self, url: str | bytes, **kwargs: Any) -> requests.Response:
        """Fail with a connection error for the first few calls.

        Args:
            url: Requested URL.
            **kwargs: Request options.

        Returns:
            The canned hub response once the failures are used up.

        Raises:
            requests.ConnectionError: While failures remain.
        """
        self.calls += 1
        if self.calls <= self.failures:
            msg = "Temporary failure in name resolution"
            raise requests.ConnectionError(msg)
        return super().head(url, **kwargs)


def test_resolve_retries_transient_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream.time, "sleep", lambda _s: None)  # no real waiting
    session = _FlakySession(failures=3)
    remote = stream.Remote("c", "train/S2L2A/x.tar", session=session, token="")
    assert session.calls == 4
    assert remote.size == 123


def test_resolve_gives_up_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stream.time, "sleep", lambda _s: None)
    session = _FlakySession(failures=1000)
    with pytest.raises(OSError, match="could not resolve"):
        stream.Remote("c", "train/S2L2A/x.tar", session=session, token="")

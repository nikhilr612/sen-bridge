# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Tests of the weight download: digest checks and the offline fallback."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from sen_bridge import weights

if TYPE_CHECKING:
    from pathlib import Path

_PAYLOAD = b"not really weights"

pytestmark = pytest.mark.usefixtures("_cache")


@pytest.fixture
def _cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the weight cache at an empty directory."""
    monkeypatch.setenv("SEN_BRIDGE_CACHE", str(tmp_path))


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> list[str]:
    """Replace the downloader with one that writes ``payload``.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        payload: Bytes to write for every URL.

    Returns:
        The list the fake appends each requested URL to.
    """
    urls: list[str] = []

    def fake(url: str, dst: Path) -> None:
        urls.append(url)
        dst.write_bytes(payload)

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake)
    return urls


def test_download_is_verified_and_pinned_to_the_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = hashlib.sha256(_PAYLOAD).hexdigest()
    monkeypatch.setitem(weights.CONVERTED, "fid_inception", digest)
    urls = _serve(monkeypatch, _PAYLOAD)
    out = weights.path("fid_inception")
    assert out.read_bytes() == _PAYLOAD
    assert urls == [f"{weights._HUB}fid_inception.safetensors"]
    assert weights.HUB_REVISION in urls[0]


def test_digest_mismatch_raises_and_leaves_no_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve(monkeypatch, b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        weights.path("lpips_vgg")
    assert not list(weights.cache_dir().glob("lpips_vgg*"))


def test_cached_file_is_not_downloaded_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (weights.cache_dir() / "fid_inception.safetensors").write_bytes(b"x")
    urls = _serve(monkeypatch, b"y")
    weights.path("fid_inception")
    assert urls == []


def test_failed_download_falls_back_to_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def offline(_url: str, _dst: Path) -> None:
        msg = "offline"
        raise OSError(msg)

    calls: list[bool] = []

    def convert() -> None:
        calls.append(True)
        (weights.cache_dir() / "lpips_vgg.safetensors").write_bytes(b"z")

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", offline)
    monkeypatch.setattr(weights, "prepare", convert)
    assert weights.path("lpips_vgg").read_bytes() == b"z"
    assert calls == [True]

# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Fetch and verify the weights the metrics need, as safetensors.

The PyTorch and JAX implementations load exactly the same numbers. On first use the
two converted files are downloaded from a pinned revision of
``nikhilr01/sen-bridge-weights`` and checked against pinned SHA-256 digests; no
torch is needed for that. If the download fails, the files are rebuilt from the
three original hosts (also hash-pinned), which needs torch once to unpickle the
original ``.pth`` files. See NOTICE for the licence of each source; the VGG16 part
carries no licence assertion.

Run ``python -m sen_bridge.weights`` to fetch everything ahead of time.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20
_HTTPS = "https://"

#: Immutable revision of the converted weights on the Hub.
HUB_REVISION = "1761854f6d9e6aac2e918a3bff18f89a377db1ce"
_HUB = (
    f"https://huggingface.co/nikhilr01/sen-bridge-weights/resolve/{HUB_REVISION}/"
)

#: Pinned SHA-256 digests of the converted safetensors files.
CONVERTED: dict[str, str] = {
    "fid_inception": (
        "b7a528bf4051f62c057bc14657865abf180a23eb1748d00ff0933d84698202ed"
    ),
    "lpips_vgg": "a96ef46bc0b0a064b67e277c15af82caa42278dd79aba0034815ed355e1aaacb",
}


@dataclass(frozen=True)
class Source:
    """One downloadable weight file.

    Attributes:
        url: HTTPS location of the original file.
        sha256: Pinned SHA-256 digest of the file.
        name_prefix: First eight hex digits of the digest if the publisher encodes
            it in the file name (the ``torch.hub`` convention), else ``None``.
    """

    url: str
    sha256: str
    name_prefix: str | None = None


SOURCES: dict[str, Source] = {
    # FID Inception-v3: the original TensorFlow inception-2015-12-05 weights ported
    # by pytorch-fid (Apache-2.0).
    "fid_inception": Source(
        url=(
            "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/"
            "pt_inception-2015-12-05-6726825d.pth"
        ),
        sha256="6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
        name_prefix="6726825d",
    ),
    # torchvision VGG16 ImageNet weights, the LPIPS trunk.
    "vgg16": Source(
        url="https://download.pytorch.org/models/vgg16-397923af.pth",
        sha256="397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
        name_prefix="397923af",
    ),
    # LPIPS v0.1 linear calibration layers for the VGG trunk (BSD-2-Clause, shipped
    # in the lpips repository). The publisher gives no digest, so this one is pinned
    # on first download.
    "lpips_lin": Source(
        url=(
            "https://raw.githubusercontent.com/richzhang/"
            "PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth"
        ),
        sha256="a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
    ),
}


def cache_dir() -> Path:
    """Return the weight cache directory, creating it if needed.

    The location is ``$SEN_BRIDGE_CACHE`` if set, else ``~/.cache/sen_bridge``.

    Returns:
        The cache directory.
    """
    directory = Path(
        os.environ.get("SEN_BRIDGE_CACHE", Path.home() / ".cache" / "sen_bridge")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sha256(path: Path) -> str:
    """Hash a file in chunks so large checkpoints are never held in memory.

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


def _fetch(name: str) -> Path:
    """Download one source if absent and verify it against its pinned digest.

    Args:
        name: Key of :data:`SOURCES`.

    Returns:
        Path of the verified raw file.

    Raises:
        ValueError: If the URL is not HTTPS, or the file does not match its pinned
            digest (the file is then removed).
    """
    src = SOURCES[name]
    if not src.url.startswith(_HTTPS):
        msg = f"{name}: refusing non-HTTPS URL {src.url}"
        raise ValueError(msg)
    dst = cache_dir() / "raw" / Path(src.url).name
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists():
        partial = dst.with_suffix(".part")
        # The HTTPS scheme was checked above.
        urllib.request.urlretrieve(  # ruff: ignore[suspicious-url-open-usage]
            src.url, partial
        )
        partial.rename(dst)
    digest = _sha256(dst)
    if digest != src.sha256 or (
        src.name_prefix is not None and not digest.startswith(src.name_prefix)
    ):
        dst.unlink()
        msg = (
            f"{name}: SHA-256 {digest} does not match the pinned digest; "
            "file removed"
        )
        raise ValueError(msg)
    return dst


def _download_converted(name: str) -> Path:
    """Download one converted file from the Hub and verify its digest.

    Args:
        name: Key of :data:`CONVERTED`.

    Returns:
        Path of the verified file in the cache.

    Raises:
        ValueError: If the downloaded file does not match its pinned digest (it
            is then removed).
    """
    dst = cache_dir() / f"{name}.safetensors"
    partial = dst.with_suffix(".part")
    # The URL is a fixed HTTPS constant.
    urllib.request.urlretrieve(  # ruff: ignore[suspicious-url-open-usage]
        f"{_HUB}{dst.name}", partial
    )
    digest = _sha256(partial)
    if digest != CONVERTED[name]:
        partial.unlink()
        msg = f"{name}: SHA-256 {digest} does not match the pinned digest; removed"
        raise ValueError(msg)
    partial.rename(dst)
    return dst


def prepare() -> None:
    """Rebuild the converted weights from the original hosts.

    This is the fallback for when the Hub copy is unavailable. It imports torch
    (extra ``convert``) to unpickle the original ``.pth`` files.
    """
    # Optional dependency (extra ``convert``) needed only by this one-off step.
    import safetensors.torch as st  # ruff: ignore[import-outside-top-level]

    # Optional dependency (extra ``convert``) needed only by this one-off step.
    import torch  # ruff: ignore[import-outside-top-level]

    fid = torch.load(_fetch("fid_inception"), map_location="cpu", weights_only=True)
    st.save_file(
        {k: v.contiguous() for k, v in fid.items()},
        cache_dir() / "fid_inception.safetensors",
    )

    vgg = torch.load(_fetch("vgg16"), map_location="cpu", weights_only=True)
    lin = torch.load(_fetch("lpips_lin"), map_location="cpu", weights_only=True)
    prefix = "features."
    tensors = {
        f"vgg.{k[len(prefix) :]}": v.contiguous()
        for k, v in vgg.items()
        if k.startswith(prefix)
    }
    tensors.update({
        k: v.contiguous() for k, v in lin.items()
    })  # lin0.model.1.weight ...
    st.save_file(tensors, cache_dir() / "lpips_vgg.safetensors")


def path(name: str) -> Path:
    """Return the converted safetensors file for a network, fetching it if missing.

    The file is downloaded from the Hub revision pinned in :data:`HUB_REVISION`. If
    the download fails (offline, host down), it is rebuilt from the original hosts
    with :func:`prepare`. A digest mismatch is never retried or hidden.

    Args:
        name: ``"fid_inception"`` or ``"lpips_vgg"``.

    Returns:
        Path of the safetensors file.
    """
    out = cache_dir() / f"{name}.safetensors"
    if out.exists():
        return out
    try:
        return _download_converted(name)
    # Offline or host down: rebuild from the original sources instead.
    except OSError:
        prepare()
    return out


def main() -> None:
    """Fetch all weights, then print the resulting files and their digests."""
    for name, digest in CONVERTED.items():
        sys.stdout.write(f"{path(name)} {path(name).stat().st_size} {digest}\n")


if __name__ == "__main__":
    main()

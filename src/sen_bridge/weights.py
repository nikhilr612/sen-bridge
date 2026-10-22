# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Download, verify and convert the three third-party weight sets the metrics need.

Nothing is redistributed with sen-bridge: files are fetched from their original
hosts on first use, checked against pinned SHA-256 digests, and stored as
safetensors so the PyTorch and JAX implementations load exactly the same numbers.
See NOTICE for the licence of each source.

Run ``python -m sen_bridge.weights`` to fetch and convert everything. Converting
needs torch once, to unpickle the original ``.pth`` files.
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


def prepare() -> None:
    """Fetch, verify and convert all weights into the cache.

    This is a one-off step. It imports torch (extra ``convert``) to unpickle the
    original ``.pth`` files.
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
    """Return the converted safetensors file for a network, building it if missing.

    Args:
        name: ``"fid_inception"`` or ``"lpips_vgg"``.

    Returns:
        Path of the safetensors file.
    """
    out = cache_dir() / f"{name}.safetensors"
    if not out.exists():
        prepare()
    return out


def main() -> None:
    """Fetch and convert all weights, then print the resulting files and digests."""
    prepare()
    for name in ("fid_inception", "lpips_vgg"):
        sys.stdout.write(f"{path(name)} {path(name).stat().st_size}\n")
    for name, src in SOURCES.items():
        sys.stdout.write(f"{name} {src.sha256}\n")


if __name__ == "__main__":
    main()

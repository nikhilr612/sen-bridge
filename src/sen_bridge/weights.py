"""Download, verify and convert the three third-party weight sets the metrics need.

Nothing is redistributed with sen-bridge: files are fetched from their original hosts on first use,
checked against pinned SHA-256 digests, and stored as safetensors so the PyTorch and JAX
implementations load exactly the same numbers. See NOTICE for the licence of each source.

    python -m sen_bridge.weights          # fetch and convert everything (needs torch once, for unpickling)
"""
import hashlib
import os
import urllib.request
from pathlib import Path

SOURCES = {
    # FID Inception-v3: the original TensorFlow inception-2015-12-05 weights ported by pytorch-fid (Apache-2.0).
    "fid_inception": dict(
        url="https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth",
        sha256="6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"),
    # torchvision VGG16 ImageNet weights, the LPIPS trunk.
    "vgg16": dict(url="https://download.pytorch.org/models/vgg16-397923af.pth",
                  sha256="397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"),
    # LPIPS v0.1 linear calibration layers for the VGG trunk (BSD-2-Clause, shipped in the lpips repository).
    "lpips_lin": dict(
        url="https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth",
        sha256="a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868"),  # pinned on first download; no published digest
}
# torch.hub convention: the file name carries the first 8 hex digits of the SHA-256.
NAME_PREFIX = {"fid_inception": "6726825d", "vgg16": "397923af"}


def cache_dir() -> Path:
    d = Path(os.environ.get("SEN_BRIDGE_CACHE", Path.home() / ".cache" / "sen_bridge"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(name: str) -> Path:
    src = SOURCES[name]
    dst = cache_dir() / "raw" / Path(src["url"]).name
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(src["url"], tmp)
        tmp.rename(dst)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    want, prefix = src["sha256"], NAME_PREFIX.get(name)
    if (want and h != want) or (prefix and not h.startswith(prefix)):
        dst.unlink()
        raise ValueError(f"{name}: SHA-256 {h} does not match the pinned digest; file removed")
    return dst


def path(name: str) -> Path:
    """Path of the converted safetensors file ('fid_inception' or 'lpips_vgg'); builds it if missing."""
    out = cache_dir() / f"{name}.safetensors"
    if not out.exists():
        prepare()
    return out


def prepare():
    """Fetch, verify and convert all weights (one-off; needs torch to unpickle the .pth files)."""
    import torch
    from safetensors.torch import save_file

    fid = torch.load(_fetch("fid_inception"), map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in fid.items()}, cache_dir() / "fid_inception.safetensors")

    vgg = torch.load(_fetch("vgg16"), map_location="cpu", weights_only=True)
    lin = torch.load(_fetch("lpips_lin"), map_location="cpu", weights_only=True)
    tensors = {f"vgg.{k[len('features.'):]}": v.contiguous() for k, v in vgg.items() if k.startswith("features.")}
    tensors.update({k: v.contiguous() for k, v in lin.items()})  # lin0.model.1.weight ... lin4.model.1.weight
    save_file(tensors, cache_dir() / "lpips_vgg.safetensors")


if __name__ == "__main__":
    prepare()
    for n in ("fid_inception", "lpips_vgg"):
        print(path(n), path(n).stat().st_size)
    for n in SOURCES:
        print(n, hashlib.sha256((cache_dir() / "raw" / Path(SOURCES[n]["url"]).name).read_bytes()).hexdigest())

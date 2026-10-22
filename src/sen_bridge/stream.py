"""Stream SSL4EO-S12 v1.1 samples straight from Hugging Face with HTTP range requests.

Nothing is written to disk except small cached tar indices. A shard is a tar of per-sample
Zarr-v2 zips; a season of S2L2A is a single blosc chunk (~1 MB), so we fetch only the
seasons that survive the metadata filters.
"""
import io
import json
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = "embed2scale/SSL4EO-S12-v1.1"
PAX_SKIP = 1536  # pax header + data block + real header, read in one request


def shard_path(commit: str, split: str, modality: str, shard: str) -> str:
    return f"datasets/{REPO}@{commit}/{split}/{modality}/{shard}"


def tar_index(fs, path: str, cache_dir: Path | None = None) -> dict:
    """{member name: (data offset, size)} by walking tar headers with ranged reads."""
    cache = None
    if cache_dir:
        cache = Path(cache_dir) / (path.replace("/", "__") + ".json")
        if cache.exists():
            return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    total = fs.info(path)["size"]
    out, off = {}, 0
    while off < total:
        blk = fs.cat_file(path, start=off, end=min(off + PAX_SKIP, total))
        ti = tarfile.TarInfo.frombuf(blk[:512], "utf-8", "strict")
        hdr = 512
        if ti.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            hdr += 512 * ((ti.size + 511) // 512) + 512
            ti = tarfile.TarInfo.frombuf(blk[hdr - 512:hdr], "utf-8", "strict")
        out[ti.name] = (off + hdr, ti.size)
        off += hdr + 512 * ((ti.size + 511) // 512)
        while off < total and fs.cat_file(path, start=off, end=off + 512) == b"\0" * 512:
            off += 512  # tar end blocks
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


class RangeView(io.RawIOBase):
    """Read-only window [offset, offset+size) of a remote file, with ranged reads."""

    def __init__(self, fs, path, offset, size):
        self.fs, self.path, self.base, self.size, self.pos = fs, path, offset, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, pos, whence=0):
        self.pos = {0: pos, 1: self.pos + pos, 2: self.size + pos}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        b = self.fs.cat_file(self.path, start=self.base + self.pos, end=self.base + self.pos + n)
        self.pos += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)


class Sample:
    """One sample's zarr-zip; read arrays chunk by chunk."""

    def __init__(self, fs, path, offset, size):
        self.zip = zipfile.ZipFile(RangeView(fs, path, offset, size))

    def attrs(self, name):
        return json.loads(self.zip.read(f"{name}/.zarray"))

    def chunk(self, array: str, index: str) -> np.ndarray:
        """Decode one chunk, e.g. chunk('bands', '1.0.0.0') -> [1,12,264,264]."""
        from numcodecs import Blosc
        meta = self.attrs(array)
        raw = Blosc().decode(self.zip.read(f"{array}/{index}"))
        return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["chunks"])

    def s2_rgb(self, season: int) -> np.ndarray:
        """B4,B3,B2 digital numbers for one season: [3,264,264] (band order B01,B02,B03,B04,...)."""
        c = self.chunk("bands", f"{season}.0.0.0")[0]
        return c[[3, 2, 1]]  # B04, B03, B02

    def s1(self, season: int) -> np.ndarray:
        return self.chunk("bands", "0.0.0.0")[season].astype(np.float32)  # [2,264,264] VV,VH dB

    def dem(self) -> np.ndarray:
        return self.chunk("bands", "0.0.0.0")[0, 0].astype(np.float32)


def prefetch_indices(fs, paths, cache_dir, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: tar_index(fs, p, cache_dir), paths))

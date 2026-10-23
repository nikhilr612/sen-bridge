# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Memory-mapped access to the SEN-Bridge ``.npy`` release.

The ``.npy`` release stores each split as raw 8-bit arrays, so reading a pair is a
slice of a memory-mapped file with no image decoding. It is a second copy of the
same data as the Parquet release (``nikhilr01/sen-bridge``); it does not replace it.

    from sen_bridge import data

    train = data.load_dataset("train")           # downloads on first use
    example = train[1234]                        # one pair
    batch = train[[5, 1, 9]]                     # a batch, any index order
    for batch in train.iter_batches(256, shuffle=True, seed=0):
        ...

Per example the keys are ``sar`` and ``optical`` (``[128, 128, 3]``, ``uint8``, or
``float32`` in ``[-1, 1]`` with ``float_images=True``), ``meta`` (``[7]``,
``float32``) and ``id``. Batches add a leading axis. A :class:`Split` has
``__len__`` and ``__getitem__``, so it is a map-style dataset for PyTorch, Grain and
anything else that indexes by position, and it pickles by path, so worker processes
reopen the memory maps instead of copying them.

The files come from a pinned revision of the Hub repository over HTTPS and are
checked against the SHA-256 digests in its manifest, which is itself pinned here.
Nothing in this module needs more than numpy.
"""

from __future__ import annotations

import contextlib
import hashlib
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sen_bridge import weights

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pandas as pd
    from numpy.typing import NDArray

    Example = dict[str, NDArray[Any]]
    Index = int | slice | Sequence[int] | NDArray[np.integer[Any]]

#: Hub dataset repository holding the ``.npy`` files.
REPO = "nikhilr01/sen-bridge-npy"
#: Immutable revision of the repository this version of the code reads.
HUB_REVISION = "0" * 40
#: SHA-256 of the manifest of that revision.
MANIFEST_SHA256 = "0" * 64
#: Split names, in the order the release lists them.
SPLITS = ("train", "validation", "test")
#: Name of the digest list that every release directory carries.
MANIFEST = "MANIFEST.sha256"
#: Per-split arrays, by key.
ARRAYS = {"sar": "sar", "optical": "optical", "meta": "meta", "id": "ids"}
#: Columns of the optional full metadata table.
METADATA_FILE = "metadata.parquet"

_CHUNK = 1 << 20
_HTTPS = "https://"
_SCALE = 127.5


def array_name(split: str, key: str) -> str:
    """Return the file name of one array of a split.

    Args:
        split: Split name.
        key: Key of :data:`ARRAYS`.

    Returns:
        The ``.npy`` file name.
    """
    return f"{split}_{ARRAYS[key]}.npy"


def _sha256(path: Path) -> str:
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


def parse_manifest(path: Path) -> dict[str, str]:
    """Read a ``sha256sum``-style digest list.

    Args:
        path: Manifest file.

    Returns:
        File name to SHA-256 digest.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if digest and name:
            out[name] = digest
    return out


def _retrieve(name: str, dst: Path, expected: str, revision: str) -> None:
    """Download one file and keep it only if its digest matches.

    Args:
        name: File name inside the repository.
        dst: Destination path.
        expected: Required SHA-256 digest.
        revision: Repository revision to read.

    Raises:
        ValueError: If the digest does not match (the download is removed).
    """
    url = f"{_HTTPS}huggingface.co/datasets/{REPO}/resolve/{revision}/{name}"
    partial = dst.with_suffix(dst.suffix + ".part")
    # The URL is built from an HTTPS constant.
    urllib.request.urlretrieve(  # ruff: ignore[suspicious-url-open-usage]
        url, partial
    )
    digest = _sha256(partial)
    if digest != expected:
        partial.unlink()
        msg = f"{name}: SHA-256 {digest} does not match the pinned digest; removed"
        raise ValueError(msg)
    partial.rename(dst)


def _verified(path: Path, expected: str) -> bool:
    """Tell whether a file is present and matches its digest.

    The (slow) hash is computed once; a sidecar ``<name>.ok`` then vouches for the
    file, so later calls only compare a short string.

    Args:
        path: File to check.
        expected: Required SHA-256 digest.

    Returns:
        ``True`` if the file is intact.
    """
    if not path.exists():
        return False
    sidecar = path.with_suffix(path.suffix + ".ok")
    if sidecar.exists() and sidecar.read_text(encoding="utf-8") == expected:
        return True
    if _sha256(path) != expected:
        return False
    # A read-only directory just means the hash is recomputed next time.
    with contextlib.suppress(OSError):
        sidecar.write_text(expected, encoding="utf-8")
    return True


def default_root() -> Path:
    """Return the default download directory (inside the weight cache).

    Returns:
        ``<cache>/data``, created if missing.
    """
    root = weights.cache_dir() / "data"
    root.mkdir(exist_ok=True)
    return root


def download(
    root: Path | str | None = None,
    splits: Sequence[str] = SPLITS,
    *,
    metadata: bool = False,
    revision: str | None = None,
) -> Path:
    """Fetch the files of some splits, verifying every one.

    Files that are already present and match their digest are not downloaded
    again.

    Args:
        root: Target directory (default: inside the sen-bridge cache).
        splits: Splits to fetch.
        metadata: Also fetch the full metadata table (read with pandas).
        revision: Repository revision; defaults to :data:`HUB_REVISION`.

    Returns:
        The directory holding the files.
    """
    target = Path(root) if root is not None else default_root()
    target.mkdir(parents=True, exist_ok=True)
    rev = revision or HUB_REVISION
    manifest = target / MANIFEST
    if not manifest.exists() or _sha256(manifest) != MANIFEST_SHA256:
        manifest.unlink(missing_ok=True)
        _retrieve(MANIFEST, manifest, MANIFEST_SHA256, rev)
    digests = parse_manifest(manifest)
    wanted = [array_name(s, k) for s in splits for k in ARRAYS]
    if metadata:
        wanted.append(METADATA_FILE)
    for name in wanted:
        dst = target / name
        if _verified(dst, digests[name]):
            continue
        dst.unlink(missing_ok=True)
        _retrieve(name, dst, digests[name], rev)
        _verified(dst, digests[name])
    return target


class Split:
    """One split of the ``.npy`` release, indexable by position.

    Attributes:
        split: Split name.
        root: Directory of the files.
        sar: Radar images, memory-mapped ``uint8`` ``[N, 128, 128, 3]``.
        optical: Optical images, memory-mapped, same shape.
        meta: Conditioning vectors, ``float32`` ``[N, 7]``, held in memory.
        ids: Example ids, ``[N]`` strings, held in memory.
        float_images: Whether examples hold ``float32`` images in ``[-1, 1]``.
    """

    def __init__(
        self,
        root: Path | str,
        split: str,
        *,
        float_images: bool = False,
    ) -> None:
        """Open the arrays of one split.

        Args:
            root: Directory of the files.
            split: Split name.
            float_images: Return ``float32`` images in ``[-1, 1]`` instead of
                ``uint8``.

        Raises:
            ValueError: If the arrays disagree on the number of examples.
        """
        self.root = Path(root)
        self.split = split
        self.float_images = float_images
        self.sar = self._open("sar", mmap=True)
        self.optical = self._open("optical", mmap=True)
        self.meta = self._open("meta", mmap=False)
        self.ids = self._open("id", mmap=False)
        sizes = {len(a) for a in (self.sar, self.optical, self.meta, self.ids)}
        if len(sizes) != 1:
            msg = f"{split}: arrays have different lengths {sorted(sizes)}"
            raise ValueError(msg)

    def _open(self, key: str, *, mmap: bool) -> NDArray[Any]:
        """Open one array of this split.

        Args:
            key: Key of :data:`ARRAYS`.
            mmap: Memory-map the file (images) instead of reading it into memory
                (the small metadata arrays).

        Returns:
            The array, read-only if memory-mapped.
        """
        path = self.root / array_name(self.split, key)
        return np.load(path, mmap_mode="r" if mmap else None)

    def __getstate__(self) -> dict[str, object]:
        """Pickle by path so worker processes reopen the memory maps.

        Returns:
            The arguments that rebuild the split.
        """
        return {
            "root": self.root,
            "split": self.split,
            "float_images": self.float_images,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reopen the memory maps after unpickling.

        Args:
            state: State from :meth:`__getstate__`.
        """
        self.__init__(
            state["root"], state["split"], float_images=state["float_images"]
        )

    def __len__(self) -> int:
        """Return the number of examples.

        Returns:
            The number of pairs in the split.
        """
        return len(self.ids)

    def _image(self, array: NDArray[Any]) -> NDArray[Any]:
        """Convert an image array to the requested format.

        Args:
            array: ``uint8`` images.

        Returns:
            The same array, or ``float32`` in ``[-1, 1]``.
        """
        if not self.float_images:
            return np.ascontiguousarray(array)
        return array.astype(np.float32) / _SCALE - 1.0

    def __getitem__(self, index: Index) -> Example:
        """Read one example or a batch.

        Args:
            index: An integer (one example), or a slice or a sequence or array
                of integers (a batch with a leading axis, in the given order).

        Returns:
            Arrays under ``sar``, ``optical``, ``meta`` and ``id``.
        """
        pos: Any = index
        if not isinstance(index, (int, np.integer, slice)):
            pos = np.asarray(index, dtype=np.int64)
        return {
            "sar": self._image(self.sar[pos]),
            "optical": self._image(self.optical[pos]),
            "meta": np.array(self.meta[pos]),
            "id": np.array(self.ids[pos]),
        }

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
        epochs: int = 1,
    ) -> Iterator[Example]:
        """Iterate over the split in batches.

        Args:
            batch_size: Examples per batch.
            shuffle: Visit examples in a random order (new order each epoch).
            seed: Seed of the shuffle.
            drop_last: Skip a final batch smaller than ``batch_size``.
            epochs: Number of passes.

        Yields:
            Batches in the format of :meth:`__getitem__`.
        """
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(self)) if shuffle else np.arange(len(self))
            for start in range(0, len(order), batch_size):
                chosen = order[start : start + batch_size]
                if drop_last and len(chosen) < batch_size:
                    break
                yield self[chosen]

    def metadata(self) -> pd.DataFrame:
        """Return the full metadata of this split as a table, in row order.

        Needs pandas and pyarrow (extra ``data``) and the file fetched with
        ``download(..., metadata=True)``.

        Returns:
            One row per example, with a ``row`` column holding the position.
        """
        # Optional dependency (extra ``data``), needed only for this method.
        import pandas as pd  # ruff: ignore[import-outside-top-level]

        table = pd.read_parquet(self.root / METADATA_FILE)
        # pandas-stubs lose the DataFrame type through boolean indexing.
        rows = table[table["split"] == self.split].reset_index(drop=True)
        # Narrow the type for the checker; pandas-stubs lose it above.
        assert isinstance(rows, pd.DataFrame)  # ruff: ignore[assert]
        return rows


def load_dataset(
    split: str | None = None,
    *,
    root: Path | str | None = None,
    float_images: bool = False,
) -> Split | dict[str, Split]:
    """Open one split (or all) of the ``.npy`` release, downloading if needed.

    Args:
        split: ``"train"``, ``"validation"`` or ``"test"``; ``None`` opens all.
        root: Directory of the files. Files are fetched into it if missing;
            default is the sen-bridge cache.
        float_images: Return ``float32`` images in ``[-1, 1]`` instead of
            ``uint8``.

    Returns:
        A :class:`Split`, or a dict of all three when ``split`` is ``None``.
    """
    names = SPLITS if split is None else (split,)
    folder = download(root, names)
    opened = {n: Split(folder, n, float_images=float_images) for n in names}
    return opened if split is None else opened[split]

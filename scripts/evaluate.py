# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Reference evaluation for SEN-Bridge.

Usage::

    python scripts/evaluate.py SAMPLES.npy --data DATASET_DIR_OR_REPO
        [--split test] [--fid-stats stats/fid_test.npz] [--framework torch|jax]
        [--bootstrap 1000] [--out results.json]

``SAMPLES.npy`` holds one generated optical image per example of the split, in
dataset order: uint8 ``[N, 128, 128, 3]`` or float in [-1, 1]. The script reports
FID (against the statistics file, or against the split's targets if none is given)
and per-image PSNR, SSIM and LPIPS as mean and std, with 95% bootstrap intervals
resampled by geographic cluster. Never use the test split to choose models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from datasets import load_dataset

if TYPE_CHECKING:
    from numpy.typing import NDArray

_HALF_RANGE = 127.5
_METRICS = ("psnr", "ssim", "lpips")
_CI = (2.5, 97.5)


class Backend(Protocol):
    """A metrics backend (PyTorch or JAX) behind a common interface."""

    def paired(
        self, gen: NDArray[Any], tgt: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Score generated images against targets.

        Args:
            gen: Generated images, uint8 or float in [-1, 1], ``[B, H, W, 3]``.
            tgt: Target images, same shape.

        Returns:
            PSNR, SSIM and LPIPS for each image.
        """
        ...

    def feed(self, images: NDArray[Any], *, real: bool) -> None:
        """Add images to the FID accumulator.

        Args:
            images: Images, uint8 or float in [-1, 1], ``[B, H, W, 3]``.
            real: Whether they are real targets.
        """
        ...

    def fid(self) -> float:
        """Compute the FID.

        Returns:
            The Frechet Inception Distance.
        """
        ...


def to_float(x: NDArray[Any]) -> NDArray[np.float32]:
    """Map images to float32 in [-1, 1].

    Args:
        x: uint8 images, or floats that are already in [-1, 1].

    Returns:
        Float32 images in [-1, 1].
    """
    if x.dtype == np.uint8:
        return x.astype(np.float32) / _HALF_RANGE - 1
    return x.astype(np.float32)


class TorchBackend:
    """Metrics from ``sen_bridge.torch``."""

    def __init__(self, stats: Path | None) -> None:
        """Create the backend.

        Args:
            stats: Real-target statistics file, or ``None`` to featurise the
                targets.
        """
        import torch  # ruff: ignore[import-outside-top-level]  (optional dependency)

        from sen_bridge import torch as sb  # ruff: ignore[import-outside-top-level]

        self._torch, self._sb = torch, sb
        self._fid = sb.FID.from_stats(stats) if stats else sb.FID()

    def _prep(self, x: NDArray[Any]) -> Any:  # ruff: ignore[any-type]  (a framework tensor)
        return self._torch.from_numpy(to_float(x)).permute(0, 3, 1, 2)

    def paired(
        self, gen: NDArray[Any], tgt: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Score generated images against targets.

        Args:
            gen: Generated images, ``[B, H, W, 3]``.
            tgt: Target images, same shape.

        Returns:
            PSNR, SSIM and LPIPS for each image.
        """
        a, b = self._prep(gen), self._prep(tgt)
        return (
            self._sb.psnr(a, b).numpy(),
            self._sb.ssim(a, b).numpy(),
            self._sb.lpips(a, b).numpy(),
        )

    def feed(self, images: NDArray[Any], *, real: bool) -> None:
        """Add images to the FID accumulator.

        Args:
            images: Images, ``[B, H, W, 3]``.
            real: Whether they are real targets.
        """
        self._fid.update(self._prep(images), real=real)

    def fid(self) -> float:
        """Compute the FID.

        Returns:
            The Frechet Inception Distance.
        """
        return self._fid.compute()


class JaxBackend:
    """Metrics from ``sen_bridge.jax``."""

    def __init__(self, stats: Path | None) -> None:
        """Create the backend.

        Args:
            stats: Real-target statistics file, or ``None`` to featurise the
                targets.
        """
        import jax  # ruff: ignore[import-outside-top-level]  (optional dependency)
        import jax.numpy as jnp  # ruff: ignore[import-outside-top-level]

        from sen_bridge import jax as sb  # ruff: ignore[import-outside-top-level]

        self._jnp = jnp
        self._fns = [jax.jit(jax.vmap(f)) for f in (sb.psnr, sb.ssim, sb.lpips)]
        self._fid = sb.FID.from_stats(stats) if stats else sb.FID()

    def paired(
        self, gen: NDArray[Any], tgt: NDArray[Any]
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Score generated images against targets.

        Args:
            gen: Generated images, ``[B, H, W, 3]``.
            tgt: Target images, same shape.

        Returns:
            PSNR, SSIM and LPIPS for each image.
        """
        a, b = self._jnp.asarray(to_float(gen)), self._jnp.asarray(to_float(tgt))
        psnr, ssim, lpips = (np.asarray(f(a, b)) for f in self._fns)
        return psnr, ssim, lpips

    def feed(self, images: NDArray[Any], *, real: bool) -> None:
        """Add images to the FID accumulator.

        Args:
            images: Images, ``[B, H, W, 3]``.
            real: Whether they are real targets.
        """
        self._fid.update(self._jnp.asarray(to_float(images)), real=real)

    def fid(self) -> float:
        """Compute the FID.

        Returns:
            The Frechet Inception Distance.
        """
        return self._fid.compute()


def cluster_bootstrap(
    values: dict[str, NDArray[Any]],
    clusters: NDArray[Any],
    n: int,
) -> dict[str, tuple[float, float]]:
    """Bootstrap 95% intervals of per-image means, resampling whole clusters.

    Args:
        values: Per-image scores by metric name.
        clusters: Cluster id of each image.
        n: Number of bootstrap resamples.

    Returns:
        The lower and upper bound for each metric.
    """
    unique = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n):
        picked = np.concatenate([
            members[c] for c in rng.choice(unique, len(unique))
        ])
        draws.append([v[picked].mean() for v in values.values()])
    low, high = np.percentile(draws, _CI, axis=0)
    return {
        k: (float(a), float(b)) for k, a, b in zip(values, low, high, strict=True)
    }


def main() -> None:
    """Evaluate a set of generated samples and print the metrics as JSON.

    Raises:
        SystemExit: If the number of samples differs from the split size.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fid-stats", type=Path)
    parser.add_argument("--framework", choices=["torch", "jax"], default="torch")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    ds = load_dataset(args.data, split=args.split)
    generated = np.load(args.samples)
    if len(generated) != len(ds):
        msg = f"{len(generated)} samples for {len(ds)} examples"
        raise SystemExit(msg)
    backend: Backend = (
        TorchBackend(args.fid_stats)
        if args.framework == "torch"
        else JaxBackend(args.fid_stats)
    )

    scores: dict[str, list[NDArray[Any]]] = {m: [] for m in _METRICS}
    for start in range(0, len(ds), args.batch):
        stop = min(start + args.batch, len(ds))
        targets = np.stack([
            np.asarray(r["optical"]) for r in ds.select(range(start, stop))
        ])
        gen = generated[start:stop]
        for name, value in zip(_METRICS, backend.paired(gen, targets), strict=True):
            scores[name].append(value)
        backend.feed(gen, real=False)
        if args.fid_stats is None:
            backend.feed(targets, real=True)
    per_image = {k: np.concatenate(v) for k, v in scores.items()}

    intervals = cluster_bootstrap(
        per_image, np.asarray(ds["cluster_id"]), args.bootstrap
    )
    result: dict[str, Any] = {"n": len(ds), "fid": backend.fid()}
    for name, values in per_image.items():
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "ci95_cluster_bootstrap": list(intervals[name]),
        }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

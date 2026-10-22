# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Framework-free FID maths and the statistics file format.

The Frechet distance follows pytorch-fid (Apache-2.0, Maximilian Seitzer); see
NOTICE.

Changes from pytorch-fid: the distance is separated from feature extraction, typed
and documented, and written for current SciPy (no ``disp`` argument).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    Matrix = NDArray[np.float64]

#: Tolerance on the imaginary part of the matrix square root, as in pytorch-fid.
_IMAG_ATOL = 1e-3


def feature_stats(feats: Matrix) -> tuple[Matrix, Matrix]:
    """Compute the mean and covariance of a feature matrix.

    Args:
        feats: Features with one row per image, shape ``[N, D]``.

    Returns:
        The mean ``[D]`` and the covariance ``[D, D]`` of the rows.
    """
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet_distance(
    mu1: Matrix,
    sigma1: Matrix,
    mu2: Matrix,
    sigma2: Matrix,
    eps: float = 1e-6,
) -> float:
    """Compute the Frechet distance between two Gaussians.

    A singular covariance product is stabilised by a small diagonal offset, as in
    pytorch-fid.

    Args:
        mu1: Mean of the first Gaussian, shape ``[D]``.
        sigma1: Covariance of the first Gaussian, shape ``[D, D]``.
        mu2: Mean of the second Gaussian, shape ``[D]``.
        sigma2: Covariance of the second Gaussian, shape ``[D, D]``.
        eps: Diagonal offset used only when the matrix square root is not finite.

    Returns:
        The squared Frechet distance.

    Raises:
        ValueError: If the matrix square root has a non-negligible imaginary
            component.
    """
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=_IMAG_ATOL):
            detail = f"imaginary component {np.max(np.abs(covmean.imag)):.3g}"
            msg = f"{detail} in the matrix square root"
            raise ValueError(msg)
        covmean = covmean.real
    return float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


def save_stats(path: Path | str, mu: Matrix, sigma: Matrix) -> None:
    """Write feature statistics to an ``.npz`` file.

    Args:
        path: Destination file.
        mu: Feature mean, shape ``[D]``.
        sigma: Feature covariance, shape ``[D, D]``.
    """
    np.savez(path, mu=mu, sigma=sigma)


def load_stats(path: Path | str) -> tuple[Matrix, Matrix]:
    """Read feature statistics written by :func:`save_stats`.

    Args:
        path: Source ``.npz`` file.

    Returns:
        The mean ``[D]`` and covariance ``[D, D]``.
    """
    with np.load(path) as data:
        return data["mu"], data["sigma"]

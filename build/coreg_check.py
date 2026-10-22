# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Estimate the radar-optical shift of each crop (20 m pixels) from PNG pairs.

Usage::

    python build/coreg_check.py PILOT_DIR [--n 400]

The check is cross-modal: it correlates gradient magnitudes of the optical luminance
and the radar VV band. A random-pair control (radar from a different crop) shows
what a meaningless correlation peak looks like.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SEARCH = 8  # search radius in pixels
_SIZE = 128
_CENTRE = _SIZE // 2
_SIGMA = 1.0
_EPS = 1e-6
_Z_EPS = 1e-9
_TEXTURED_STD = 0.06
_MAX_WATER = 0.5
_RELIABLE_QUANTILE = 0.95
_TERCILES = 3
_P90 = 0.9
_ONE_PX, _TWO_PX, _FAR_PX = 1.5, 2.5, 4


class Shift(NamedTuple):
    """The best-aligning shift of one pair.

    Attributes:
        dy: Vertical shift in pixels (positive: radar content is lower).
        dx: Horizontal shift in pixels (positive: radar content is to the right).
        peak: Normalised correlation at the best shift.
        z: Height of the peak over the search window, in standard deviations.
    """

    dy: int
    dx: int
    peak: float
    z: float


def gradient_feature(a: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute a smoothed, standardised gradient-magnitude image.

    Args:
        a: Image of shape ``[H, W]``.

    Returns:
        The standardised gradient magnitude.
    """
    smooth = gaussian_filter(a.astype(np.float32), _SIGMA)
    grad_y, grad_x = np.gradient(smooth)
    mag = gaussian_filter(np.hypot(grad_x, grad_y), _SIGMA)
    return ((mag - mag.mean()) / (mag.std() + _EPS)).astype(np.float32)


def best_shift(opt: NDArray[np.float32], sar: NDArray[np.float32]) -> Shift:
    """Find the shift that best aligns radar to optical.

    Args:
        opt: Optical luminance, shape ``[128, 128]``.
        sar: Radar VV, shape ``[128, 128]``.

    Returns:
        The best shift within the search radius and its correlation peak.
    """
    window = np.hanning(_SIZE)[:, None] * np.hanning(_SIZE)[None, :]
    a, b = gradient_feature(opt) * window, gradient_feature(sar) * window
    corr = np.fft.irfft2(np.fft.rfft2(b) * np.conj(np.fft.rfft2(a)), s=a.shape)
    corr = np.fft.fftshift(corr) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS)
    lo, hi = _CENTRE - _SEARCH, _CENTRE + _SEARCH + 1
    patch = corr[lo:hi, lo:hi]
    iy, ix = np.unravel_index(patch.argmax(), patch.shape)
    z = (patch.max() - patch.mean()) / (patch.std() + _Z_EPS)
    return Shift(int(iy) - _SEARCH, int(ix) - _SEARCH, float(patch.max()), float(z))


def load_pair(
    d: Path, sample_id: str
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Read the optical luminance and radar VV of one crop.

    Args:
        d: Pilot directory.
        sample_id: Example id.

    Returns:
        Optical luminance and radar VV, both ``[128, 128]`` float32 in 0..255.
    """
    optical = Image.open(d / "optical" / f"{sample_id}.png").convert("RGB")
    sar = np.asarray(Image.open(d / "sar" / f"{sample_id}.png"), dtype=np.float32)
    return np.asarray(optical, dtype=np.float32).mean(2), sar[..., 0]


def measure(d: Path, n: int) -> pd.DataFrame:
    """Measure the shift of textured land crops and of random mismatched pairs.

    Args:
        d: Pilot directory.
        n: Number of crops to measure.

    Returns:
        One row per crop with its shift, the control peak, slope and a ``reliable``
        flag (peak above the control's 95th percentile).
    """
    kept = pd.read_parquet(d / "pilot.parquet")
    textured = kept[
        (kept["water_frac_sar"] < _MAX_WATER) & (kept["opt_std"] > _TEXTURED_STD)
    ]
    sample = textured.sample(n, random_state=0)
    ids = [str(i) for i in sample["id"]]
    rng = np.random.default_rng(0)
    rows = []
    for i, sample_id in enumerate(ids):
        opt, sar = load_pair(d, sample_id)
        found = best_shift(opt, sar)
        other = ids[(i + 1 + int(rng.integers(len(ids) - 1))) % len(ids)]
        control = best_shift(opt, load_pair(d, other)[1])
        rows.append({
            "id": sample_id,
            "dy": found.dy,
            "dx": found.dx,
            "peak": found.peak,
            "z": found.z,
            "ctrl_peak": control.peak,
            "ctrl_z": control.z,
        })
    out = pd.DataFrame(rows).merge(
        sample[["id", "slope_std", "dt_hours", "lat"]], on="id"
    )
    out["mag"] = np.hypot(out["dy"], out["dx"])
    out["reliable"] = out["peak"] > out["ctrl_peak"].quantile(_RELIABLE_QUANTILE)
    return cast("pd.DataFrame", out)  # pandas stubs lose the type


def report(r: pd.DataFrame) -> None:
    """Print the summary of a measurement.

    Args:
        r: Output of :func:`measure`.
    """
    threshold = r["ctrl_peak"].quantile(_RELIABLE_QUANTILE)
    print(
        f"n={len(r)}  control 95th-percentile peak={threshold:.3f}  "
        f"reliable (peak above it): {r['reliable'].mean():.0%}"
    )
    g = r[r["reliable"]]
    print(
        f"reliable crops: shift |d| in 20 m px  median {g['mag'].median():.1f}  "
        f"p90 {g['mag'].quantile(_P90):.1f}   "
        f"|dx| med {g['dx'].abs().median():.1f}  "
        f"|dy| med {g['dy'].abs().median():.1f}"
    )
    within1, within2 = (g["mag"] <= _ONE_PX).mean(), (g["mag"] <= _TWO_PX).mean()
    print(
        f"fraction of reliable with |d|<=1px: {within1:.0%}  <=2px: {within2:.0%}  "
        f">4px: {(g['mag'] > _FAR_PX).mean():.0%}"
    )
    print(
        f"mean signed shift dx {g['dx'].mean():.2f} dy {g['dy'].mean():.2f} "
        "(systematic offset if far from 0)"
    )
    terciles = pd.qcut(g["slope_std"], _TERCILES, labels=["flat", "mid", "steep"])
    by_slope = g.assign(slope=terciles).groupby("slope", observed=True)["mag"]
    print(
        by_slope
        .agg(["count", "median", lambda x: x.quantile(_P90)])
        .round(2)
        .to_string()
    )


def main() -> None:
    """Measure the shift on a pilot directory and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()
    result = measure(args.pilot_dir, args.n)
    report(result)
    result.to_parquet(args.pilot_dir / "coreg.parquet")


if __name__ == "__main__":
    main()

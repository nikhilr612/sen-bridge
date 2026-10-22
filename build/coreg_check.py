"""Estimate radar<->optical shift per crop (in 20 m pixels) from saved PNG pairs.

Cross-modal: correlate gradient magnitudes of optical luminance and radar VV. A random-pair
control (radar from a different crop) shows what a meaningless peak looks like.
Usage: python build/coreg_check.py PILOT_DIR [--n 400]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

R = 8  # search radius in pixels


def feat(a):
    a = gaussian_filter(a.astype(np.float32), 1.0)
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    g = gaussian_filter(g, 1.0)
    return (g - g.mean()) / (g.std() + 1e-6)


def shift(opt, sar):
    """Return (dy, dx, peak, z) maximising correlation within +-R; positive = sar content is shifted down/right."""
    w = np.hanning(128)[:, None] * np.hanning(128)[None, :]
    A, B = feat(opt) * w, feat(sar) * w
    cc = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(B) * np.conj(np.fft.rfft2(A)), s=A.shape))
    cc /= (np.linalg.norm(A) * np.linalg.norm(B) + 1e-6)
    c = 64
    win = cc[c - R:c + R + 1, c - R:c + R + 1]
    iy, ix = np.unravel_index(win.argmax(), win.shape)
    z = (win.max() - win.mean()) / (win.std() + 1e-9)
    return iy - R, ix - R, float(win.max()), float(z)


def load(d, id_):
    opt = np.asarray(Image.open(d / "optical" / f"{id_}.png").convert("RGB"), dtype=np.float32).mean(2)
    sar = np.asarray(Image.open(d / "sar" / f"{id_}.png"), dtype=np.float32)[..., 0]  # VV
    return opt, sar


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pilot_dir"); ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args(); d = Path(a.pilot_dir)
    k = pd.read_parquet(d / "pilot.parquet")
    k = k[(k.water_frac_sar < 0.5) & (k.opt_std > 0.06)].sample(a.n, random_state=0)  # textured land only
    ids = list(k.id); rng = np.random.default_rng(0)
    rows = []
    for i, id_ in enumerate(ids):
        opt, sar = load(d, id_)
        dy, dx, pk, z = shift(opt, sar)
        _, sar2 = load(d, ids[(i + 1 + int(rng.integers(len(ids) - 1))) % len(ids)])
        cy, cx, cpk, cz = shift(opt, sar2)
        rows.append(dict(id=id_, dy=dy, dx=dx, peak=pk, z=z, ctrl_peak=cpk, ctrl_z=cz))
    r = pd.DataFrame(rows).merge(k[["id", "slope_std", "dt_hours", "lat"]], on="id")
    r["mag"] = np.hypot(r.dy, r.dx)
    thr = r.ctrl_peak.quantile(0.95)
    r["reliable"] = r.peak > thr
    print(f"n={len(r)}  control 95th-percentile peak={thr:.3f}  reliable (peak above it): {r.reliable.mean():.0%}")
    g = r[r.reliable]
    print("reliable crops: shift |d| in 20 m px  median %.1f  p90 %.1f   |dx| med %.1f  |dy| med %.1f"
          % (g.mag.median(), g.mag.quantile(.9), g.dx.abs().median(), g.dy.abs().median()))
    print("fraction of reliable with |d|<=1px: %.0f%%  <=2px: %.0f%%  >4px: %.0f%%" % (
        (g.mag <= 1.5).mean() * 100, (g.mag <= 2.5).mean() * 100, (g.mag > 4).mean() * 100))
    print("mean signed shift dx %.2f dy %.2f (systematic offset if far from 0)" % (g.dx.mean(), g.dy.mean()))
    g = g.assign(slope=pd.qcut(g.slope_std, 3, labels=["flat", "mid", "steep"]))
    print(g.groupby("slope", observed=True).mag.agg(["count", "median", lambda x: x.quantile(.9)]).round(2).to_string())
    r.to_parquet(d / "coreg.parquet")


if __name__ == "__main__":
    main()

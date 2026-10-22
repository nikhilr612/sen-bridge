# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Write Inception statistics of the real optical targets.

Usage::

    python scripts/make_fid_stats.py DATASET_DIR_OR_REPO OUT_DIR

Writes ``OUT_DIR/fid_validation.npz`` and ``OUT_DIR/fid_test.npz``, computed with
the sen-bridge PyTorch FID so ``FID.from_stats`` can score without featurising the
reals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from sen_bridge import torch as sb

_BATCH = 200
_HALF_RANGE = 127.5
_SPLITS = (("validation", "fid_validation"), ("test", "fid_test"))


def main() -> None:
    """Compute and save the statistics of the validation and test targets."""
    data, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for split, name in _SPLITS:
        ds = load_dataset(data, split=split)
        fid = sb.FID()
        for start in range(0, len(ds), _BATCH):
            stop = min(start + _BATCH, len(ds))
            rows = ds.select(range(start, stop))
            images = np.stack([np.asarray(r["optical"]) for r in rows])
            tensor = torch.from_numpy(images.astype(np.float32) / _HALF_RANGE - 1)
            fid.update(tensor.permute(0, 3, 1, 2), real=True)
        fid.save_stats(out / f"{name}.npz", real=True)
        print(split, len(ds), "->", out / f"{name}.npz")


if __name__ == "__main__":
    main()

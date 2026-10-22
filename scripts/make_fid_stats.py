"""Write Inception mean/covariance of the real optical targets: stats/fid_validation.npz and stats/fid_test.npz.

    python scripts/make_fid_stats.py DATASET_DIR_OR_REPO OUT_DIR
Computed with the sen-bridge PyTorch FID so `FID.from_stats` can score without featurising the reals.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from sen_bridge import torch as sb

data, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for split, name in (("validation", "fid_validation"), ("test", "fid_test")):
    ds = load_dataset(data, split=split)
    fid = sb.FID()
    for i in range(0, len(ds), 200):
        x = np.stack([np.asarray(r["optical"]) for r in ds.select(range(i, min(i + 200, len(ds))))])
        fid.update(torch.from_numpy(x.astype(np.float32) / 127.5 - 1).permute(0, 3, 1, 2), True)
    fid.save_stats(out / f"{name}.npz", True)
    print(split, len(ds), "->", out / f"{name}.npz")

"""Reference evaluation for SEN-Bridge.

    python scripts/evaluate.py SAMPLES.npy --data DATASET_DIR_OR_REPO [--split test] [--fid-stats stats/fid_test.npz]
                               [--framework torch|jax] [--bootstrap 1000] [--out results.json]

SAMPLES.npy: one generated optical image per test example, in dataset order: uint8 [N, 128, 128, 3] (or float in [-1, 1]).
Reports FID (against the stats file, or against the test targets if none is given), and per-image PSNR / SSIM / LPIPS
as mean and std, with 95% bootstrap intervals resampled by geographic cluster. Never use the test split to choose models.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples"); ap.add_argument("--data", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--fid-stats"); ap.add_argument("--framework", choices=["torch", "jax"], default="torch")
    ap.add_argument("--bootstrap", type=int, default=1000); ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(a.data, split=a.split)
    gen = np.load(a.samples)
    assert len(gen) == len(ds), f"{len(gen)} samples for {len(ds)} examples"
    to_float = lambda x: (x.astype(np.float32) / 127.5 - 1) if x.dtype == np.uint8 else x.astype(np.float32)
    tgt = lambda sl: np.stack([np.asarray(r["optical"]) for r in ds.select(range(*sl))])

    if a.framework == "torch":
        import torch
        from sen_bridge import torch as sb
        fid = sb.FID.from_stats(a.fid_stats) if a.fid_stats else sb.FID()
        prep = lambda x: torch.from_numpy(to_float(x)).permute(0, 3, 1, 2)
        paired = lambda x, y: [m(prep(x), prep(y)).numpy() for m in (sb.psnr, sb.ssim, sb.lpips)]
        feed = lambda x, real: fid.update(prep(x), real)
    else:
        import jax
        import jax.numpy as jnp
        from sen_bridge import jax as sb
        fid = sb.FID.from_stats(a.fid_stats) if a.fid_stats else sb.FID()
        fns = [jax.jit(jax.vmap(f)) for f in (sb.psnr, sb.ssim, sb.lpips)]
        paired = lambda x, y: [np.asarray(f(jnp.asarray(to_float(x)), jnp.asarray(to_float(y)))) for f in fns]
        feed = lambda x, real: fid.update(jnp.asarray(to_float(x)), real)

    cols = [[], [], []]
    for i in range(0, len(ds), a.batch):
        sl = (i, min(i + a.batch, len(ds)))
        t, g = tgt(sl), gen[sl[0]:sl[1]]
        for c, v in zip(cols, paired(g, t)):
            c.append(v)
        feed(g, False)
        if not a.fid_stats:
            feed(t, True)
    psnr, ssim, lp = (np.concatenate(c) for c in cols)

    clusters = np.asarray(ds["cluster_id"])
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.bootstrap):
        pick = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq))])
        boots.append([m[pick].mean() for m in (psnr, ssim, lp)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    res = {"n": len(ds), "fid": fid.compute()}
    for name, m, l, h in zip(("psnr", "ssim", "lpips"), (psnr, ssim, lp), lo, hi):
        res[name] = {"mean": float(m.mean()), "std": float(m.std()), "ci95_cluster_bootstrap": [float(l), float(h)]}
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()

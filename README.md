# sen-bridge

FID, LPIPS, PSNR and SSIM with the exact settings the SEN-Bridge benchmark specifies, in PyTorch and JAX, plus the
code that builds the SEN-Bridge dataset (radar-to-optical Sentinel-1/2 pairs for flow-matching and bridge models).

The metrics package knows nothing about splits, loaders or models: you decide what to compare and when.

```
pip install "sen-bridge[torch]"     # or [jax]
python -m sen_bridge.weights        # one-off: downloads, hash-checks and converts the three weight files (needs torch)
```

## Metrics

Inputs are float images in **[-1, 1]**, 3 channels, 128x128. `sen_bridge.torch` is channels-first, `sen_bridge.jax` channels-last.
Every metric first rounds to uint8, so scores never depend on the float precision of your outputs.

```python
from sen_bridge import torch as sb               # x, y: [B, 3, H, W]
sb.psnr(x, y), sb.ssim(x, y), sb.lpips(x, y)     # each -> [B] (one value per image: average, bootstrap or slice them yourself)

fid = sb.FID.from_stats("stats/fid_test.npz")    # real statistics ship with the dataset
fid.update(generated, real=False)                # batches; resized on the CPU
score = fid.compute()
```

```python
from sen_bridge import jax as sb                 # single image [H, W, C]; batch with jit + vmap
psnr = jax.jit(jax.vmap(sb.psnr))(x, y)          # likewise ssim, lpips
fid = sb.FID(); fid.update(images_nhwc, real=True); ...
```

Frozen settings (none are arguments): FID uses the pytorch-fid Inception-v3 weights and clean-fid's CPU bicubic resize to 299x299;
LPIPS uses the VGG trunk with the v0.1 linear layers; SSIM uses an 11x11 Gaussian, sigma 1.5, data range 1; PSNR has data range 1.

**Same weights, same numbers.** Both backends load one set of safetensors files. The test suite checks: LPIPS and Inception features
identical to the `lpips` and `pytorch-fid` packages, PSNR and SSIM within 1e-4 of scikit-image, and JAX versus PyTorch within 1e-4 on
LPIPS and 0.01 on FID. (FID on very few images, where the covariance is singular, can differ slightly more; use thousands of images.)

## Evaluating a model

```
python scripts/evaluate.py samples.npy --data <dataset> --split test --fid-stats stats/fid_test.npz
```
`samples.npy` holds one generated optical image per test example in dataset order. The script reports FID and
per-image PSNR/SSIM/LPIPS with 95% bootstrap intervals resampled by geographic cluster. Choose checkpoints on validation only.

## Licences

Code: Apache-2.0 (see `LICENSE`, `NOTICE`, `licenses/`). Weights are **not** redistributed; they come from their original hosts and are
SHA-256 checked. The FID and LPIPS-linear weights are Apache-2.0 and BSD-2-Clause. The VGG16 trunk has no published weights licence;
see `docs/LICENSES.md` before commercial use.

## Layout

- `src/sen_bridge/` metrics only (numpy, scipy, Pillow; torch or jax by extra).
- `src/sen_bridge_build/` dataset build helpers; `build/` the pipeline scripts; `docs/` audit and licence notes.

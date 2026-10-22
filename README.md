# sen-bridge

FID, LPIPS, PSNR and SSIM with the exact settings the SEN-Bridge benchmark specifies, in PyTorch and JAX, plus the
code that builds the SEN-Bridge dataset (radar-to-optical Sentinel-1/2 pairs for flow-matching and bridge models).

The metrics package knows nothing about splits, loaders or models: you decide what to compare and when.

```
pip install "sen-bridge[torch]"     # or "sen-bridge[jax,convert]" for JAX
python -m sen_bridge.weights        # one-off: download, hash-check and convert the weights
```

The base install needs only numpy, scipy and Pillow. Optional dependencies are extras:

| Extra | Installs | For |
|---|---|---|
| `torch` | torch, torchvision, safetensors | `sen_bridge.torch` |
| `jax` | jax, safetensors | `sen_bridge.jax` |
| `convert` | torch, safetensors | the one-off weight conversion (needed even with `jax`) |
| `eval` | datasets | `scripts/evaluate.py`, `scripts/make_fid_stats.py` |
| `build` | pandas, pyarrow, requests, huggingface_hub, datasets, pyyaml, numcodecs | the dataset build |
| `dev` | everything above plus ruff, ty, pytest, stubs | `./check.sh` |

## Metrics

Inputs are float images in **[-1, 1]**, 3 channels, 128x128. `sen_bridge.torch` is channels-first, `sen_bridge.jax` channels-last.
Every metric first rounds to uint8, so scores never depend on the float precision of your outputs.

```python
from sen_bridge import torch as sb  # x, y: [B, 3, H, W]

(
    sb.psnr(x, y),
    sb.ssim(x, y),
    sb.lpips(x, y),
)  # each -> [B] (one value per image: average, bootstrap or slice them yourself)

fid = sb.FID.from_stats(
    "stats/fid_test.npz"
)  # real statistics ship with the dataset
fid.update(generated, real=False)  # batches; resized on the CPU
score = fid.compute()
```

```python
from sen_bridge import jax as sb  # single image [H, W, C]; batch with jit + vmap

psnr = jax.jit(jax.vmap(sb.psnr))(x, y)  # likewise ssim, lpips
fid = sb.FID()
fid.update(images_nhwc, real=True)
...
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

- **Code:** Apache-2.0, Copyright 2026 Nikhil Raghavendra (`LICENSE`, `NOTICE`). Adapted
  third-party code and its licences are in `NOTICE` and `licenses/`.
- **Dataset:** CC BY 4.0, the licence of its source, SSL4EO-S12 v1.1. Attribution and the
  Copernicus notice are in the dataset card and `NOTICE`.
- **Weights** are **not** redistributed; they come from their original hosts and are
  SHA-256 checked. The FID and LPIPS-linear weights are Apache-2.0 and BSD-2-Clause.
  The VGG16 trunk (used by LPIPS) has no published weights licence upstream;
  `docs/LICENSES.md` records the position.

## Development standards

One fixed standard, enforced by the tools rather than by convention:

- **Formatting:** `ruff format`, 84 columns. `ruff format --check` must pass.
- **Lint:** `ruff check` with `select = ["ALL"]` and preview rules. Only two rules are
  disabled, both because they conflict with the formatter; per-file exceptions
  (`assert` in tests, `print` in command-line scripts) are written down in
  `pyproject.toml`. Docstrings and comments obey the same 84-column limit.
- **Documentation:** Google-style docstrings on every module, class, function and
  method, checked by the `D` (pydocstyle, `convention = "google"`) and `DOC`
  (pydoclint) rules, so `Args`, `Returns`, `Raises` and `Attributes` must match the
  code.
- **Types:** `ty check` with every rule set to `error`; all code is fully annotated.
- **All at once:** `./check.sh` (also available as a pre-commit config).

## Layout

- `src/sen_bridge/` metrics only (numpy, scipy, Pillow; torch or jax by extra).
- `src/sen_bridge_build/` dataset build helpers; `build/` the pipeline scripts; `docs/` audit and licence notes.

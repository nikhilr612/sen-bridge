"""JAX metrics: FID, LPIPS (VGG), PSNR, SSIM, numerically matched to sen_bridge.torch.

Inputs are float images in [-1, 1], channels-last. `psnr`, `ssim` and `lpips` are pure single-image functions

Evaluation only: rounding to uint8 has zero gradient, so never use these metrics as a training loss. Values outside
[-1, 1] are clipped silently (a model outputting 2.0 scores like one outputting 1.0), so check your output range. Rounding
is idempotent on 8-bit images (u / 127.5 - 1 -> same u), and 0.0 falls on a rounding tie and becomes ~+0.004; both are
inherent to the 8-bit encoding and identical in the two backends.
[H, W, C] -> scalar (jit-safe); use `jax.jit(jax.vmap(fn))` for batches. `FID.update` takes [B, H, W, C] batches
because it accumulates state and resizes on the CPU. Same frozen settings and the same safetensors weights as the
PyTorch module; the Inception graph follows pytorch-fid (Apache-2.0, see NOTICE) and LPIPS follows
richzhang/PerceptualSimilarity (BSD-2-Clause). Written as plain functions over a parameter dict, so only jax is needed.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from safetensors.numpy import load_file

from . import weights
from .fid_core import feature_stats, frechet_distance, load_stats, save_stats
from .resize import resize_batch

HI = lax.Precision.HIGHEST


def quantise(x):
    """Round to uint8 levels; returns floats in [0, 1]."""
    return jnp.round((jnp.clip(x, -1, 1) + 1) * 127.5) / 255.0


def psnr(x, y):
    """[H, W, C] -> scalar, data range 1."""
    mse = jnp.mean((quantise(x) - quantise(y)) ** 2)
    return 10 * jnp.log10(1.0 / jnp.maximum(mse, 1e-10))


def ssim(x, y):
    """[H, W, C] -> scalar; 11x11 Gaussian, sigma 1.5, valid-region mean over channels."""
    x, y = quantise(x)[None], quantise(y)[None]
    c = x.shape[-1]
    g = jnp.exp(-((jnp.arange(11, dtype=jnp.float32) - 5) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = jnp.tile((g[:, None] * g[None, :])[:, :, None, None], (1, 1, 1, c))  # HWIO, depthwise
    f = lambda a: lax.conv_general_dilated(a, w, (1, 1), "VALID", feature_group_count=c,
                                           dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=HI)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return jnp.mean(s)


def _conv(x, w, stride=1, pad=(0, 0)):
    return lax.conv_general_dilated(x, w, (stride, stride), ((pad[0], pad[0]), (pad[1], pad[1])),
                                    dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=HI)


def _pool(x, kind, k=3, stride=1, pad=0, count_pad=True):
    win, strd = (1, k, k, 1), (1, stride, stride, 1)
    padding = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    if kind == "max":
        return lax.reduce_window(x, -jnp.inf, lax.max, win, strd, padding)
    s = lax.reduce_window(x, 0.0, lax.add, win, strd, padding)
    if count_pad:
        return s / (k * k)
    n = lax.reduce_window(jnp.ones((1,) + x.shape[1:3] + (1,), x.dtype), 0.0, lax.add, win, strd, padding)
    return s / n


# ---- LPIPS (VGG) ---------------------------------------------------------------------------------------------

_VGG_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]
_VGG_CONV_IDX = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
_VGG_TAPS = (1, 3, 6, 9, 12)  # conv ordinal after which relu1_2 ... relu5_3 are read


@lru_cache(maxsize=1)
def _lpips_params():
    with jax.ensure_compile_time_eval():  # concrete arrays even if first called inside jit (else tracers leak into the cache)
        return _load_lpips()


def _load_lpips():
    sd = load_file(str(weights.path("lpips_vgg")))
    convs = [(jnp.asarray(sd[f"vgg.{i}.weight"].transpose(2, 3, 1, 0)), jnp.asarray(sd[f"vgg.{i}.bias"])) for i in _VGG_CONV_IDX]
    lins = [jnp.asarray(sd[f"lin{i}.model.1.weight"][0, :, 0, 0]) for i in range(5)]
    return dict(convs=convs, lins=lins)


def _vgg_feats(p, x):
    outs, ci = [], 0
    for item in _VGG_CFG:
        if item == "M":
            x = _pool(x, "max", k=2, stride=2)
        else:
            w, b = p["convs"][ci]
            x = jax.nn.relu(_conv(x, w, pad=(1, 1)) + b)
            if ci in _VGG_TAPS:
                outs.append(x)
            ci += 1
    return outs


_SHIFT = jnp.array([-0.030, -0.088, -0.188])
_SCALE = jnp.array([0.458, 0.448, 0.450])


def _lpips_apply(p, x, y):
    x, y = ((a - _SHIFT) / _SCALE for a in (x, y))
    val = 0.0
    for fx, fy, lin in zip(_vgg_feats(p, x), _vgg_feats(p, y), p["lins"]):
        fx, fy = (f / (jnp.sqrt(jnp.sum(f ** 2, axis=-1, keepdims=True)) + 1e-10) for f in (fx, fy))
        val = val + jnp.mean(jnp.sum((fx - fy) ** 2 * lin, axis=-1), axis=(1, 2))
    return val


def lpips(x, y):
    """[H, W, C] -> scalar LPIPS-VGG distance."""
    q = lambda t: (quantise(t) * 2 - 1)[None]
    return _lpips_apply(_lpips_params(), q(x), q(y))[0]


# ---- FID Inception -------------------------------------------------------------------------------------------

_NO_PAD = {"Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_4a_3x3", "Mixed_6a.branch3x3", "Mixed_6a.branch3x3dbl_3",
           "Mixed_7a.branch3x3_2", "Mixed_7a.branch7x7x3_4"}
_STRIDE2 = {"Conv2d_1a_3x3", "Mixed_6a.branch3x3", "Mixed_6a.branch3x3dbl_3", "Mixed_7a.branch3x3_2", "Mixed_7a.branch7x7x3_4"}


@lru_cache(maxsize=1)
def _inception_params():
    with jax.ensure_compile_time_eval():
        return _load_inception()


def _load_inception():
    sd = load_file(str(weights.path("fid_inception")))
    names = sorted({k.rsplit(".conv.", 1)[0] for k in sd if k.endswith(".conv.weight")})
    p = {}
    for n in names:
        scale = sd[f"{n}.bn.weight"] / np.sqrt(sd[f"{n}.bn.running_var"] + 1e-3)  # BasicConv2d: BatchNorm eps=0.001
        p[n] = dict(w=jnp.asarray(sd[f"{n}.conv.weight"].transpose(2, 3, 1, 0)), scale=jnp.asarray(scale),
                    shift=jnp.asarray(sd[f"{n}.bn.bias"] - sd[f"{n}.bn.running_mean"] * scale))
    return p


def _bc(p, name, x):
    q = p[name]
    kh, kw = q["w"].shape[:2]
    pad = (0, 0) if name in _NO_PAD else ((kh - 1) // 2, (kw - 1) // 2)
    y = _conv(x, q["w"], 2 if name in _STRIDE2 else 1, pad)
    return jax.nn.relu(y * q["scale"] + q["shift"])


def _block_a(p, n, x):
    b = lambda s, h: _bc(p, f"{n}.{s}", h)
    pool = b("branch_pool", _pool(x, "avg", 3, 1, 1, count_pad=False))
    return jnp.concatenate([b("branch1x1", x), b("branch5x5_2", b("branch5x5_1", x)),
                            b("branch3x3dbl_3", b("branch3x3dbl_2", b("branch3x3dbl_1", x))), pool], -1)


def _block_b(p, n, x):
    b = lambda s, h: _bc(p, f"{n}.{s}", h)
    return jnp.concatenate([b("branch3x3", x), b("branch3x3dbl_3", b("branch3x3dbl_2", b("branch3x3dbl_1", x))),
                            _pool(x, "max", 3, 2)], -1)


def _block_c(p, n, x):
    b = lambda s, h: _bc(p, f"{n}.{s}", h)
    d = x
    for s in ("branch7x7dbl_1", "branch7x7dbl_2", "branch7x7dbl_3", "branch7x7dbl_4", "branch7x7dbl_5"):
        d = b(s, d)
    pool = b("branch_pool", _pool(x, "avg", 3, 1, 1, count_pad=False))
    return jnp.concatenate([b("branch1x1", x), b("branch7x7_3", b("branch7x7_2", b("branch7x7_1", x))), d, pool], -1)


def _block_d(p, n, x):
    b = lambda s, h: _bc(p, f"{n}.{s}", h)
    d = x
    for s in ("branch7x7x3_1", "branch7x7x3_2", "branch7x7x3_3", "branch7x7x3_4"):
        d = b(s, d)
    return jnp.concatenate([b("branch3x3_2", b("branch3x3_1", x)), d, _pool(x, "max", 3, 2)], -1)


def _block_e(p, n, x, pool):
    b = lambda s, h: _bc(p, f"{n}.{s}", h)
    t = b("branch3x3_1", x)
    t = jnp.concatenate([b("branch3x3_2a", t), b("branch3x3_2b", t)], -1)
    d = b("branch3x3dbl_2", b("branch3x3dbl_1", x))
    d = jnp.concatenate([b("branch3x3dbl_3a", d), b("branch3x3dbl_3b", d)], -1)
    pooled = _pool(x, "avg", 3, 1, 1, count_pad=False) if pool == "avg" else _pool(x, "max", 3, 1, 1)
    return jnp.concatenate([b("branch1x1", x), t, d, b("branch_pool", pooled)], -1)


def _inception_apply(p, x):
    """[B, 299, 299, 3] in [-1, 1] -> [B, 2048]."""
    x = _bc(p, "Conv2d_3b_1x1", _pool(_bc(p, "Conv2d_2b_3x3", _bc(p, "Conv2d_2a_3x3", _bc(p, "Conv2d_1a_3x3", x))), "max", 3, 2))
    x = _pool(_bc(p, "Conv2d_4a_3x3", x), "max", 3, 2)
    for n in ("5b", "5c", "5d"):
        x = _block_a(p, f"Mixed_{n}", x)
    x = _block_b(p, "Mixed_6a", x)
    for n in ("6b", "6c", "6d", "6e"):
        x = _block_c(p, f"Mixed_{n}", x)
    x = _block_d(p, "Mixed_7a", x)
    x = _block_e(p, "Mixed_7b", x, "avg")
    x = _block_e(p, "Mixed_7c", x, "max")
    return jnp.mean(x, axis=(1, 2))


_inception_jit = jax.jit(_inception_apply)


class FID:
    """Accumulator over real and generated images: update(images, real), compute(), reset()."""

    def __init__(self):
        self._real_stats = None
        self.reset()

    @classmethod
    def from_stats(cls, path):
        self = cls()
        self._real_stats = load_stats(path)
        return self

    def reset(self):
        self._real, self._fake = [], []

    def features(self, images) -> np.ndarray:
        u8 = np.asarray(jnp.round((jnp.clip(images, -1, 1) + 1) * 127.5).astype(jnp.uint8))
        x = jnp.asarray(resize_batch(u8)) / 127.5 - 1.0  # CPU bicubic, then back to the device
        return np.asarray(_inception_jit(_inception_params(), x), dtype=np.float64)

    def update(self, images, real: bool):
        (self._real if real else self._fake).append(self.features(images))

    def stats(self, real: bool):
        return feature_stats(np.concatenate(self._real if real else self._fake))

    def save_stats(self, path, real: bool = True):
        save_stats(path, *self.stats(real))

    def compute(self) -> float:
        real = self._real_stats or self.stats(True)
        return frechet_distance(*real, *self.stats(False))

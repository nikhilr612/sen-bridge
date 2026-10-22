# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""JAX metrics: FID, LPIPS (VGG), PSNR, SSIM, matched to ``sen_bridge.torch``.

Inputs are float images in [-1, 1], channels-last. ``psnr``, ``ssim`` and ``lpips``
are pure single-image functions ``[H, W, C] -> scalar`` (jit-safe); use
``jax.jit(jax.vmap(fn))`` for batches. ``FID.update`` takes ``[B, H, W, C]`` batches
because it accumulates state and resizes on the CPU. Every metric first rounds to
uint8.

Evaluation only: rounding to uint8 has zero gradient, so never use these metrics as
a training loss. Values outside [-1, 1] are clipped silently (a model outputting 2.0
scores like one outputting 1.0), so check your output range. Rounding is idempotent
on 8-bit images (``u / 127.5 - 1`` maps back to the same ``u``), and 0.0 falls on a
rounding tie and becomes about +0.004; both are inherent to the 8-bit encoding. The
uint8 levels are identical in the two backends (the float values can differ by one
ulp).

Same frozen settings and the same safetensors weights as the PyTorch module; the
Inception graph follows pytorch-fid (Apache-2.0, see NOTICE) and LPIPS follows
richzhang/PerceptualSimilarity (BSD-2-Clause). Written as plain functions over a
parameter dict, so only jax is needed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from safetensors.numpy import load_file

from sen_bridge import weights
from sen_bridge.fid_core import (
    feature_stats,
    frechet_distance,
    load_stats,
    save_stats,
)
from sen_bridge.resize import resize_batch

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_PRECISION = lax.Precision.HIGHEST
_HALF_RANGE = 127.5
_MAX_LEVEL = 255.0
_MSE_FLOOR = 1e-10
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_LPIPS_EPS = 1e-10
_BN_EPS = 1e-3  # torchvision BasicConv2d uses BatchNorm eps=0.001
_DIMS = ("NHWC", "HWIO", "NHWC")


def quantise(x: jax.Array) -> jax.Array:
    """Round images to uint8 levels.

    Args:
        x: Images in [-1, 1]; values outside are clipped.

    Returns:
        Float images in [0, 1] on the 256 uint8 levels.
    """
    return jnp.round((jnp.clip(x, -1, 1) + 1) * _HALF_RANGE) / _MAX_LEVEL


def psnr(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute the peak signal-to-noise ratio with data range 1.

    Args:
        x: One image in [-1, 1], shape ``[H, W, C]``.
        y: One image in [-1, 1], same shape as ``x``.

    Returns:
        PSNR in dB as a scalar.
    """
    mse = jnp.mean((quantise(x) - quantise(y)) ** 2)
    return 10 * jnp.log10(1.0 / jnp.maximum(mse, _MSE_FLOOR))


def ssim(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute SSIM with an 11x11 Gaussian window, sigma 1.5, data range 1.

    The score is the mean over the valid region (no padding) and the channels.

    Args:
        x: One image in [-1, 1], shape ``[H, W, C]``.
        y: One image in [-1, 1], same shape as ``x``.

    Returns:
        SSIM as a scalar.
    """
    xq, yq = quantise(x)[None], quantise(y)[None]
    channels = xq.shape[-1]
    offsets = jnp.arange(_SSIM_WINDOW, dtype=jnp.float32) - _SSIM_WINDOW // 2
    gauss = jnp.exp(-(offsets**2) / (2 * _SSIM_SIGMA**2))
    gauss /= gauss.sum()
    window = jnp.tile(
        (gauss[:, None] * gauss[None, :])[:, :, None, None], (1, 1, 1, channels)
    )

    def blur(a: jax.Array) -> jax.Array:
        return lax.conv_general_dilated(
            a,
            window,
            (1, 1),
            "VALID",
            feature_group_count=channels,
            dimension_numbers=_DIMS,
            precision=_PRECISION,
        )

    mx, my = blur(xq), blur(yq)
    sxx, syy, sxy = (
        blur(xq * xq) - mx**2,
        blur(yq * yq) - my**2,
        blur(xq * yq) - mx * my,
    )
    c1, c2 = _SSIM_K1**2, _SSIM_K2**2
    score = ((2 * mx * my + c1) * (2 * sxy + c2)) / (
        (mx**2 + my**2 + c1) * (sxx + syy + c2)
    )
    return jnp.mean(score)


def _conv(
    x: jax.Array, w: jax.Array, stride: int = 1, pad: tuple[int, int] = (0, 0)
) -> jax.Array:
    """Convolve an NHWC batch with an HWIO kernel at full precision.

    Args:
        x: Input ``[B, H, W, C]``.
        w: Kernel ``[kh, kw, in, out]``.
        stride: Stride in both spatial dimensions.
        pad: Symmetric ``(height, width)`` padding.

    Returns:
        The convolved batch.
    """
    return lax.conv_general_dilated(
        x,
        w,
        (stride, stride),
        ((pad[0], pad[0]), (pad[1], pad[1])),
        dimension_numbers=_DIMS,
        precision=_PRECISION,
    )


def _max_pool(x: jax.Array, k: int = 3, stride: int = 1, pad: int = 0) -> jax.Array:
    """Max pool an NHWC batch.

    Args:
        x: Input ``[B, H, W, C]``.
        k: Window size.
        stride: Stride.
        pad: Symmetric padding (padded cells never win the maximum).

    Returns:
        The pooled batch.
    """
    padding = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    return cast(
        "jax.Array",
        lax.reduce_window(
            x, -jnp.inf, lax.max, (1, k, k, 1), (1, stride, stride, 1), padding
        ),
    )


def _avg_pool(
    x: jax.Array,
    k: int = 3,
    stride: int = 1,
    pad: int = 0,
    *,
    count_pad: bool = True,
) -> jax.Array:
    """Average pool an NHWC batch.

    Args:
        x: Input ``[B, H, W, C]``.
        k: Window size.
        stride: Stride.
        pad: Symmetric padding.
        count_pad: Whether padded zeros count in the average (False matches
            TensorFlow).

    Returns:
        The pooled batch.
    """
    window, strides = (1, k, k, 1), (1, stride, stride, 1)
    padding = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    total = cast(
        "jax.Array", lax.reduce_window(x, 0.0, lax.add, window, strides, padding)
    )
    if count_pad:
        return total / (k * k)
    ones = jnp.ones((1, *x.shape[1:3], 1), x.dtype)
    return total / cast(
        "jax.Array", lax.reduce_window(ones, 0.0, lax.add, window, strides, padding)
    )


# ---- LPIPS (VGG) -------------------------------------------------------------

_VGG_CFG: tuple[int | Literal["M"], ...] = (
    *(
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ),
)
_VGG_CONV_IDX = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
_VGG_TAPS = (
    1,
    3,
    6,
    9,
    12,
)  # conv ordinal after which relu1_2 ... relu5_3 are read
_SHIFT = jnp.array([-0.030, -0.088, -0.188])
_SCALE = jnp.array([0.458, 0.448, 0.450])


class _LpipsParams(TypedDict):
    """Parameters of the LPIPS network.

    Attributes:
        convs: ``(kernel HWIO, bias)`` for each of the 13 VGG16 convolutions.
        lins: Channel weights of the five linear layers.
    """

    convs: list[tuple[jax.Array, jax.Array]]
    lins: list[jax.Array]


def _load_lpips() -> _LpipsParams:
    """Read the LPIPS parameters from the converted safetensors file.

    Returns:
        The VGG16 trunk and linear-layer parameters.
    """
    state = load_file(str(weights.path("lpips_vgg")))
    convs = [
        (
            jnp.asarray(state[f"vgg.{i}.weight"].transpose(2, 3, 1, 0)),
            jnp.asarray(state[f"vgg.{i}.bias"]),
        )
        for i in _VGG_CONV_IDX
    ]
    lins = [
        jnp.asarray(state[f"lin{i}.model.1.weight"][0, :, 0, 0])
        for i in range(len(_VGG_TAPS))
    ]
    return {"convs": convs, "lins": lins}


@lru_cache(maxsize=1)
def _lpips_params() -> _LpipsParams:
    """Return the LPIPS parameters, loaded once.

    Loading runs in compile-time-eval mode so the cache always holds concrete arrays
    even when the first call happens inside ``jit`` (otherwise tracers would leak
    into the cache).

    Returns:
        The VGG16 trunk and linear-layer parameters.
    """
    with jax.ensure_compile_time_eval():
        return _load_lpips()


def _vgg_feats(params: _LpipsParams, x: jax.Array) -> list[jax.Array]:
    """Return the five tapped VGG16 feature maps.

    Args:
        params: LPIPS parameters.
        x: Scaled images ``[B, H, W, 3]``.

    Returns:
        Feature maps after relu1_2 ... relu5_3.
    """
    taps: list[jax.Array] = []
    ordinal = 0
    for item in _VGG_CFG:
        if item == "M":
            x = _max_pool(x, k=2, stride=2)
            continue
        w, b = params["convs"][ordinal]
        x = jax.nn.relu(_conv(x, w, pad=(1, 1)) + b)
        if ordinal in _VGG_TAPS:
            taps.append(x)
        ordinal += 1
    return taps


def _unit_norm(feats: jax.Array) -> jax.Array:
    """Scale each feature vector to unit length across channels, as LPIPS does.

    Args:
        feats: Feature map ``[B, H, W, C]``.

    Returns:
        The normalised feature map.
    """
    return feats / (
        jnp.sqrt(jnp.sum(feats**2, axis=-1, keepdims=True)) + _LPIPS_EPS
    )


def _lpips_apply(params: _LpipsParams, x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute LPIPS for batches already in the model's [-1, 1] range.

    Args:
        params: LPIPS parameters.
        x: Images ``[B, H, W, 3]``.
        y: Images ``[B, H, W, 3]``.

    Returns:
        Distance for each pair, shape ``[B]``.
    """
    xs, ys = (x - _SHIFT) / _SCALE, (y - _SHIFT) / _SCALE
    total = jnp.zeros(xs.shape[0])
    for fx, fy, lin in zip(
        _vgg_feats(params, xs), _vgg_feats(params, ys), params["lins"], strict=True
    ):
        diff = (_unit_norm(fx) - _unit_norm(fy)) ** 2
        total += jnp.mean(jnp.sum(diff * lin, axis=-1), axis=(1, 2))
    return total


def lpips(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute the LPIPS-VGG distance between two images (lower means more similar).

    Args:
        x: One image in [-1, 1], shape ``[H, W, C]``.
        y: One image in [-1, 1], same shape as ``x``.

    Returns:
        The distance as a scalar.
    """

    def to_model_range(t: jax.Array) -> jax.Array:
        return cast(
            "jax.Array", (quantise(t) * 2 - 1)[None]
        )  # indexing is untyped for ty

    distance = _lpips_apply(_lpips_params(), to_model_range(x), to_model_range(y))
    return cast("jax.Array", distance[0])  # indexing is untyped for ty


# ---- FID Inception -----------------------------------------------------------

#: Layers whose convolution is unpadded; every other layer pads by ``k // 2``
#: (``"same"``).
_NO_PAD = frozenset({
    "Conv2d_1a_3x3",
    "Conv2d_2a_3x3",
    "Conv2d_4a_3x3",
    "Mixed_6a.branch3x3",
    "Mixed_6a.branch3x3dbl_3",
    "Mixed_7a.branch3x3_2",
    "Mixed_7a.branch7x7x3_4",
})
_STRIDE2 = frozenset({
    "Conv2d_1a_3x3",
    "Mixed_6a.branch3x3",
    "Mixed_6a.branch3x3dbl_3",
    "Mixed_7a.branch3x3_2",
    "Mixed_7a.branch7x7x3_4",
})


class _ConvBn(TypedDict):
    """One folded convolution plus batch-norm layer.

    Attributes:
        w: Kernel in HWIO layout.
        scale: Batch-norm scale, ``gamma / sqrt(var + eps)``.
        shift: Batch-norm shift, ``beta - mean * scale``.
    """

    w: jax.Array
    scale: jax.Array
    shift: jax.Array


_InceptionParams = dict[str, _ConvBn]


def _load_inception() -> _InceptionParams:
    """Read the FID Inception parameters from the converted safetensors file.

    Returns:
        Folded convolution and batch-norm parameters by layer name.
    """
    state = load_file(str(weights.path("fid_inception")))
    names = sorted({
        k.rsplit(".conv.", 1)[0] for k in state if k.endswith(".conv.weight")
    })
    params: _InceptionParams = {}
    for name in names:
        scale = state[f"{name}.bn.weight"] / np.sqrt(
            state[f"{name}.bn.running_var"] + _BN_EPS
        )
        params[name] = {
            "w": jnp.asarray(state[f"{name}.conv.weight"].transpose(2, 3, 1, 0)),
            "scale": jnp.asarray(scale),
            "shift": jnp.asarray(
                state[f"{name}.bn.bias"] - state[f"{name}.bn.running_mean"] * scale
            ),
        }
    return params


@lru_cache(maxsize=1)
def _inception_params() -> _InceptionParams:
    """Return the FID Inception parameters, loaded once (concrete under ``jit``).

    Returns:
        Folded convolution and batch-norm parameters by layer name.
    """
    with jax.ensure_compile_time_eval():
        return _load_inception()


def _bc(params: _InceptionParams, name: str, x: jax.Array) -> jax.Array:
    """Apply one convolution, batch norm and ReLU.

    Args:
        params: Inception parameters.
        name: Layer name, for example ``"Mixed_5b.branch1x1"``.
        x: Input ``[B, H, W, C]``.

    Returns:
        The activated output.
    """
    layer = params[name]
    kh, kw = layer["w"].shape[:2]
    pad = (0, 0) if name in _NO_PAD else ((kh - 1) // 2, (kw - 1) // 2)
    y = _conv(x, layer["w"], 2 if name in _STRIDE2 else 1, pad)
    return jax.nn.relu(y * layer["scale"] + layer["shift"])


def _chain(
    params: _InceptionParams, prefix: str, names: tuple[str, ...], x: jax.Array
) -> jax.Array:
    """Apply a sequence of layers of one block in order.

    Args:
        params: Inception parameters.
        prefix: Block name, for example ``"Mixed_6b"``.
        names: Layer names within the block, applied left to right.
        x: Input ``[B, H, W, C]``.

    Returns:
        The output of the last layer.
    """
    for name in names:
        x = _bc(params, f"{prefix}.{name}", x)
    return x


def _avg_pool_branch(
    params: _InceptionParams, prefix: str, x: jax.Array
) -> jax.Array:
    """Apply TensorFlow's padding-excluding 3x3 average pool, then ``branch_pool``.

    Args:
        params: Inception parameters.
        prefix: Block name.
        x: Input ``[B, H, W, C]``.

    Returns:
        The pooled branch output.
    """
    return _bc(
        params, f"{prefix}.branch_pool", _avg_pool(x, 3, 1, 1, count_pad=False)
    )


def _block_a(params: _InceptionParams, name: str, x: jax.Array) -> jax.Array:
    """Run an Inception-A block (Mixed_5b/5c/5d).

    Args:
        params: Inception parameters.
        name: Block name.
        x: Input ``[B, H, W, C]``.

    Returns:
        Concatenated branch outputs.
    """
    branches = [
        _bc(params, f"{name}.branch1x1", x),
        _chain(params, name, ("branch5x5_1", "branch5x5_2"), x),
        _chain(
            params, name, ("branch3x3dbl_1", "branch3x3dbl_2", "branch3x3dbl_3"), x
        ),
        _avg_pool_branch(params, name, x),
    ]
    return jnp.concatenate(branches, -1)


def _block_b(params: _InceptionParams, name: str, x: jax.Array) -> jax.Array:
    """Run the Inception-B reduction block (Mixed_6a).

    Args:
        params: Inception parameters.
        name: Block name.
        x: Input ``[B, H, W, C]``.

    Returns:
        Concatenated branch outputs.
    """
    branches = [
        _bc(params, f"{name}.branch3x3", x),
        _chain(
            params, name, ("branch3x3dbl_1", "branch3x3dbl_2", "branch3x3dbl_3"), x
        ),
        _max_pool(x, 3, 2),
    ]
    return jnp.concatenate(branches, -1)


def _block_c(params: _InceptionParams, name: str, x: jax.Array) -> jax.Array:
    """Run an Inception-C block (Mixed_6b..6e).

    Args:
        params: Inception parameters.
        name: Block name.
        x: Input ``[B, H, W, C]``.

    Returns:
        Concatenated branch outputs.
    """
    dbl = tuple(f"branch7x7dbl_{i}" for i in range(1, 6))
    branches = [
        _bc(params, f"{name}.branch1x1", x),
        _chain(params, name, ("branch7x7_1", "branch7x7_2", "branch7x7_3"), x),
        _chain(params, name, dbl, x),
        _avg_pool_branch(params, name, x),
    ]
    return jnp.concatenate(branches, -1)


def _block_d(params: _InceptionParams, name: str, x: jax.Array) -> jax.Array:
    """Run the Inception-D reduction block (Mixed_7a).

    Args:
        params: Inception parameters.
        name: Block name.
        x: Input ``[B, H, W, C]``.

    Returns:
        Concatenated branch outputs.
    """
    x7 = tuple(f"branch7x7x3_{i}" for i in range(1, 5))
    branches = [
        _chain(params, name, ("branch3x3_1", "branch3x3_2"), x),
        _chain(params, name, x7, x),
        _max_pool(x, 3, 2),
    ]
    return jnp.concatenate(branches, -1)


def _block_e(
    params: _InceptionParams,
    name: str,
    x: jax.Array,
    pool: Literal["avg", "max"],
) -> jax.Array:
    """Run an Inception-E block (Mixed_7b/7c).

    Args:
        params: Inception parameters.
        name: Block name.
        x: Input ``[B, H, W, C]``.
        pool: Pooling of the last branch: ``"avg"`` (padding excluded) or ``"max"``.

    Returns:
        Concatenated branch outputs.
    """
    b3 = _bc(params, f"{name}.branch3x3_1", x)
    b3 = jnp.concatenate(
        [
            _bc(params, f"{name}.branch3x3_2a", b3),
            _bc(params, f"{name}.branch3x3_2b", b3),
        ],
        -1,
    )
    dbl = _chain(params, name, ("branch3x3dbl_1", "branch3x3dbl_2"), x)
    dbl = jnp.concatenate(
        [
            _bc(params, f"{name}.branch3x3dbl_3a", dbl),
            _bc(params, f"{name}.branch3x3dbl_3b", dbl),
        ],
        -1,
    )
    pooled = (
        _avg_pool(x, 3, 1, 1, count_pad=False)
        if pool == "avg"
        else _max_pool(x, 3, 1, 1)
    )
    return jnp.concatenate(
        [
            _bc(params, f"{name}.branch1x1", x),
            b3,
            dbl,
            _bc(params, f"{name}.branch_pool", pooled),
        ],
        -1,
    )


def _inception_apply(params: _InceptionParams, x: jax.Array) -> jax.Array:
    """Extract pooled Inception features.

    Args:
        params: Inception parameters.
        x: Images in [-1, 1], shape ``[B, 299, 299, 3]``.

    Returns:
        Features of shape ``[B, 2048]``.
    """
    for stem in ("Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3"):
        x = _bc(params, stem, x)
    x = _bc(params, "Conv2d_3b_1x1", _max_pool(x, 3, 2))
    x = _max_pool(_bc(params, "Conv2d_4a_3x3", x), 3, 2)
    for name in ("5b", "5c", "5d"):
        x = _block_a(params, f"Mixed_{name}", x)
    x = _block_b(params, "Mixed_6a", x)
    for name in ("6b", "6c", "6d", "6e"):
        x = _block_c(params, f"Mixed_{name}", x)
    x = _block_d(params, "Mixed_7a", x)
    x = _block_e(params, "Mixed_7b", x, "avg")
    x = _block_e(params, "Mixed_7c", x, "max")
    return jnp.mean(x, axis=(1, 2))


_inception_jit = jax.jit(_inception_apply)


class FID:
    """Accumulator over real and generated images.

    Call :meth:`update` with batches of each kind, then :meth:`compute`. Real
    statistics can be loaded from a file with :meth:`from_stats` instead of
    featurising the real images.
    """

    def __init__(self) -> None:
        """Create an empty accumulator."""
        self._real_stats: tuple[NDArray[np.float64], NDArray[np.float64]] | None = (
            None
        )
        self._real: list[NDArray[np.float64]] = []
        self._fake: list[NDArray[np.float64]] = []

    @classmethod
    def from_stats(cls, path: Path | str) -> FID:
        """Create an accumulator whose real statistics come from a file.

        Args:
            path: ``.npz`` file written by :meth:`save_stats`.

        Returns:
            An accumulator that only needs generated images.
        """
        fid = cls()
        fid._real_stats = load_stats(path)
        return fid

    def reset(self) -> None:
        """Discard all accumulated features (loaded real statistics are kept)."""
        self._real, self._fake = [], []

    @staticmethod
    def features(images: jax.Array) -> NDArray[np.float64]:
        """Compute Inception features.

        Images are rounded to uint8, resized on the CPU with clean-fid's bicubic
        filter, then run through the network.

        Args:
            images: Images in [-1, 1], shape ``[B, H, W, 3]``.

        Returns:
            Features of shape ``[B, 2048]`` in float64.
        """
        u8 = np.asarray(
            jnp.round((jnp.clip(images, -1, 1) + 1) * _HALF_RANGE).astype(jnp.uint8)
        )
        x = jnp.asarray(resize_batch(u8)) / _HALF_RANGE - 1.0
        return np.asarray(_inception_jit(_inception_params(), x), dtype=np.float64)

    def update(self, images: jax.Array, *, real: bool) -> None:
        """Add a batch of images.

        Args:
            images: Images in [-1, 1], shape ``[B, H, W, 3]``.
            real: Whether the images are real targets (``True``) or generated
                (``False``).
        """
        (self._real if real else self._fake).append(self.features(images))

    def stats(
        self, *, real: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute the mean and covariance of accumulated features.

        Args:
            real: Use the real features (``True``) or the generated ones
                (``False``).

        Returns:
            The feature mean ``[2048]`` and covariance ``[2048, 2048]``.
        """
        return feature_stats(np.concatenate(self._real if real else self._fake))

    def save_stats(self, path: Path | str, *, real: bool = True) -> None:
        """Write accumulated feature statistics to an ``.npz`` file.

        Args:
            path: Destination file.
            real: Save the real statistics (``True``) or the generated ones
                (``False``).
        """
        save_stats(path, *self.stats(real=real))

    def compute(self) -> float:
        """Compute the FID between the real and generated images.

        Returns:
            The Frechet Inception Distance.
        """
        real = self._real_stats or self.stats(real=True)
        return frechet_distance(*real, *self.stats(real=False))

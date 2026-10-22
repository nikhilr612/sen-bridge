# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch metrics: FID, LPIPS (VGG), PSNR, SSIM.

Inputs are float images in [-1, 1], shape ``[B, 3, H, W]``. Every metric first
rounds to uint8, so scores never depend on the float precision of the outputs.

Evaluation only: rounding to uint8 has zero gradient, so never use these metrics as
a training loss. Values outside [-1, 1] are clipped silently (a model outputting 2.0
scores like one outputting 1.0), so check your output range. Rounding is idempotent
on 8-bit images (``u / 127.5 - 1`` maps back to the same ``u``), and 0.0 falls on a
rounding tie and becomes about +0.004; both are inherent to the 8-bit encoding. The
uint8 levels are identical in the two backends (the float values can differ by one
ulp).

Frozen settings (none are arguments): FID uses the pytorch-fid Inception-v3 weights
with clean-fid CPU bicubic resize to 299x299; LPIPS uses the VGG16 trunk with the
v0.1 linear layers; SSIM uses an 11x11 Gaussian, sigma 1.5, data range 1.

The FID Inception block variants below are vendored from pytorch-fid (Apache-2.0,
Copyright Maximilian Seitzer), see NOTICE. LPIPS follows
richzhang/PerceptualSimilarity (BSD-2-Clause).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
import torch.nn.functional as torch_f
from safetensors.torch import load_file
from torch import nn
from torchvision.models import vgg16
from torchvision.models.inception import (
    BasicConv2d,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)
from typing_extensions import override

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

_HALF_RANGE = 127.5
_MAX_LEVEL = 255.0
_MSE_FLOOR = 1e-10
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_LPIPS_EPS = 1e-10
_POOL_KERNEL = 3
_POOL_STRIDE = 2


def _call(module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
    """Call a module and narrow the result to a tensor.

    ``nn.Module.__call__`` is typed as returning ``Any``; every network here returns
    a tensor.

    Args:
        module: The module to call.
        *args: Tensor arguments forwarded to the module.

    Returns:
        The module's output.
    """
    return cast("torch.Tensor", module(*args))


def _unit_norm(feats: torch.Tensor) -> torch.Tensor:
    """Scale each feature vector to unit length across channels, as LPIPS does.

    Args:
        feats: Feature map ``[B, C, H, W]``.

    Returns:
        The normalised feature map.
    """
    return feats / (feats.pow(2).sum(1, keepdim=True).sqrt() + _LPIPS_EPS)


def quantise(x: torch.Tensor) -> torch.Tensor:
    """Round images to uint8 levels.

    Args:
        x: Images in [-1, 1]; values outside are clipped.

    Returns:
        Float images in [0, 1] on the 256 uint8 levels. The result has zero
        gradient.
    """
    return torch.round((x.clamp(-1, 1) + 1) * _HALF_RANGE) / _MAX_LEVEL


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the peak signal-to-noise ratio with data range 1.

    Args:
        x: Images in [-1, 1], shape ``[B, 3, H, W]``.
        y: Images in [-1, 1], same shape as ``x``.

    Returns:
        PSNR in dB for each image, shape ``[B]``.
    """
    mse = ((quantise(x) - quantise(y)) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(_MSE_FLOOR))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute SSIM with an 11x11 Gaussian window, sigma 1.5, data range 1.

    The score is the mean over the valid region (no padding) and the three channels.

    Args:
        x: Images in [-1, 1], shape ``[B, 3, H, W]``.
        y: Images in [-1, 1], same shape as ``x``.

    Returns:
        SSIM for each image, shape ``[B]``.
    """
    x, y = quantise(x), quantise(y)
    channels = x.shape[1]
    offsets = (
        torch.arange(_SSIM_WINDOW, dtype=torch.float32, device=x.device)
        - _SSIM_WINDOW // 2
    )
    gauss = torch.exp(-(offsets**2) / (2 * _SSIM_SIGMA**2))
    gauss /= gauss.sum()
    window = (
        (gauss[:, None] * gauss[None, :]).expand(channels, 1, -1, -1).contiguous()
    )

    def blur(a: torch.Tensor) -> torch.Tensor:
        return torch_f.conv2d(a, window, groups=channels)

    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = _SSIM_K1**2, _SSIM_K2**2
    score = ((2 * mx * my + c1) * (2 * sxy + c2)) / (
        (mx**2 + my**2 + c1) * (sxx + syy + c2)
    )
    return score.flatten(1).mean(1)


class _LPIPS(nn.Module):
    """LPIPS with a VGG16 trunk and the v0.1 linear layers, in evaluation mode."""

    SHIFT = (-0.030, -0.088, -0.188)
    SCALE = (0.458, 0.448, 0.450)
    #: Indices in ``vgg16().features`` of relu1_2, relu2_2, relu3_3, relu4_3 and
    #: relu5_3.
    TAPS = (3, 8, 15, 22, 29)
    CHANNELS = (64, 128, 256, 512, 512)
    #: Number of leading ``vgg16().features`` layers up to and including relu5_3.
    TRUNK_LAYERS = 30

    shift: torch.Tensor
    scale: torch.Tensor

    def __init__(self) -> None:
        """Load the trunk and linear layers from the converted safetensors file."""
        super().__init__()
        self.vgg = vgg16(weights=None).features[: self.TRUNK_LAYERS]
        state = load_file(weights.path("lpips_vgg"))
        self.vgg.load_state_dict({
            k[len("vgg.") :]: v for k, v in state.items() if k.startswith("vgg.")
        })
        convs = [nn.Conv2d(c, 1, 1, bias=False) for c in self.CHANNELS]
        for i, conv in enumerate(convs):
            conv.weight.data.copy_(state[f"lin{i}.model.1.weight"])
        self.lins = nn.ModuleList(convs)
        self.register_buffer("shift", torch.tensor(self.SHIFT).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor(self.SCALE).view(1, 3, 1, 1))
        self.eval().requires_grad_(requires_grad=False)

    def _feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the five tapped VGG feature maps of a scaled image batch.

        Args:
            x: Images in [-1, 1], shape ``[B, 3, H, W]``.

        Returns:
            Feature maps after relu1_2 ... relu5_3.
        """
        x = (x - self.shift) / self.scale
        taps: list[torch.Tensor] = []
        for i, layer in enumerate(self.vgg):
            x = _call(layer, x)
            if i in self.TAPS:
                taps.append(x)
        return taps

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """Compute the LPIPS distance.

        Args:
            x: Images in [-1, 1], shape ``[B, 3, H, W]``.
            y: Images in [-1, 1], same shape as ``x``.

        Returns:
            Distance for each pair, shape ``[B]``.
        """
        total = torch.zeros(x.shape[0], device=x.device)
        for fx, fy, lin in zip(
            self._feats(x), self._feats(y), self.lins, strict=True
        ):
            diff = (_unit_norm(fx) - _unit_norm(fy)) ** 2
            total += _call(lin, diff).mean((2, 3)).flatten()
        return total


_lpips_nets: dict[torch.device, _LPIPS] = {}


def lpips(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute LPIPS-VGG distances (lower means more similar).

    Args:
        x: Images in [-1, 1], shape ``[B, 3, H, W]``.
        y: Images in [-1, 1], same shape as ``x``.

    Returns:
        Distance for each pair, shape ``[B]``.
    """
    net = _lpips_nets.get(x.device)
    if net is None:
        net = _lpips_nets[x.device] = _LPIPS().to(x.device)

    def to_model_range(t: torch.Tensor) -> torch.Tensor:
        return quantise(t) * 2 - 1

    with (
        torch.no_grad(),
        torch.backends.cudnn.flags(enabled=True, allow_tf32=False),
    ):
        return _call(net, to_model_range(x), to_model_range(y))


class _FIDInceptionA(InceptionA):
    """Inception-A with a padding-excluding average pool, as in TensorFlow."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            x: Feature map ``[B, C, H, W]``.

        Returns:
            Concatenated branch outputs.
        """
        branches = [
            self.branch1x1(x),
            self.branch5x5_2(self.branch5x5_1(x)),
            self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
            self.branch_pool(
                torch_f.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
            ),
        ]
        return torch.cat(branches, 1)


class _FIDInceptionC(InceptionC):
    """Inception-C with a padding-excluding average pool, as in TensorFlow."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            x: Feature map ``[B, C, H, W]``.

        Returns:
            Concatenated branch outputs.
        """
        b7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        dbl = self.branch7x7dbl_1(x)
        for layer in (
            self.branch7x7dbl_2,
            self.branch7x7dbl_3,
            self.branch7x7dbl_4,
            self.branch7x7dbl_5,
        ):
            dbl = layer(dbl)
        pool = self.branch_pool(
            torch_f.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        )
        return torch.cat([self.branch1x1(x), b7, dbl, pool], 1)


class _FIDInceptionE(InceptionE):
    """Inception-E with the pooling variant used by the FID graph.

    The last block of the FID Inception uses max pooling, an artefact of the
    original TensorFlow graph.
    """

    def __init__(self, in_channels: int, pool: Literal["avg", "max"]) -> None:
        """Create the block.

        Args:
            in_channels: Number of input channels.
            pool: ``"avg"`` (padding excluded) or ``"max"``.
        """
        super().__init__(in_channels)
        self.pool = pool

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block.

        Args:
            x: Feature map ``[B, C, H, W]``.

        Returns:
            Concatenated branch outputs.
        """
        b3 = self.branch3x3_1(x)
        b3 = torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1)
        dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        dbl = torch.cat([self.branch3x3dbl_3a(dbl), self.branch3x3dbl_3b(dbl)], 1)
        if self.pool == "avg":
            pooled = torch_f.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        else:
            pooled = torch_f.max_pool2d(x, 3, 1, 1)
        return torch.cat([self.branch1x1(x), b3, dbl, self.branch_pool(pooled)], 1)


class _FIDInception(nn.Module):
    """The FID Inception-v3, returning the 2048-d pooled features."""

    def __init__(self) -> None:
        """Build the graph and load the converted pytorch-fid weights."""
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = _FIDInceptionA(192, pool_features=32)
        self.Mixed_5c = _FIDInceptionA(256, pool_features=64)
        self.Mixed_5d = _FIDInceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = _FIDInceptionC(768, channels_7x7=128)
        self.Mixed_6c = _FIDInceptionC(768, channels_7x7=160)
        self.Mixed_6d = _FIDInceptionC(768, channels_7x7=160)
        self.Mixed_6e = _FIDInceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = _FIDInceptionE(1280, "avg")
        self.Mixed_7c = _FIDInceptionE(2048, "max")
        state = {
            k: v
            for k, v in load_file(weights.path("fid_inception")).items()
            if not k.startswith("fc.")
        }
        self.load_state_dict(state)
        self.eval().requires_grad_(requires_grad=False)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        """Extract pooled Inception features.

        Args:
            x: Images in [-1, 1], shape ``[B, 3, 299, 299]``.

        Returns:
            Features of shape ``[B, 2048]``.
        """
        for stem in (self.Conv2d_1a_3x3, self.Conv2d_2a_3x3, self.Conv2d_2b_3x3):
            x = _call(stem, x)
        x = _call(
            self.Conv2d_3b_1x1, torch_f.max_pool2d(x, _POOL_KERNEL, _POOL_STRIDE)
        )
        x = torch_f.max_pool2d(
            _call(self.Conv2d_4a_3x3, x), _POOL_KERNEL, _POOL_STRIDE
        )
        for block in (
            self.Mixed_5b,
            self.Mixed_5c,
            self.Mixed_5d,
            self.Mixed_6a,
            self.Mixed_6b,
            self.Mixed_6c,
            self.Mixed_6d,
            self.Mixed_6e,
            self.Mixed_7a,
            self.Mixed_7b,
            self.Mixed_7c,
        ):
            x = _call(block, x)
        return x.mean((2, 3))


class FID:
    """Accumulator over real and generated images.

    Call :meth:`update` with batches of each kind, then :meth:`compute`. Real
    statistics can be loaded from a file with :meth:`from_stats` instead of
    featurising the real images.

    Attributes:
        device: Device that runs the Inception network.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        """Create an empty accumulator.

        Args:
            device: Device for the Inception network. Defaults to CUDA if available,
                else CPU.
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: _FIDInception | None = None
        self._real_stats: tuple[NDArray[np.float64], NDArray[np.float64]] | None = (
            None
        )
        self._real: list[NDArray[np.float64]] = []
        self._fake: list[NDArray[np.float64]] = []

    @classmethod
    def from_stats(
        cls, path: Path | str, device: str | torch.device | None = None
    ) -> FID:
        """Create an accumulator whose real statistics come from a file.

        Args:
            path: ``.npz`` file written by :meth:`save_stats`.
            device: Device for the Inception network.

        Returns:
            An accumulator that only needs generated images.
        """
        fid = cls(device)
        fid._real_stats = load_stats(path)
        return fid

    def reset(self) -> None:
        """Discard all accumulated features (loaded real statistics are kept)."""
        self._real, self._fake = [], []

    def features(self, images: torch.Tensor) -> NDArray[np.float64]:
        """Compute Inception features.

        Images are rounded to uint8, resized on the CPU with clean-fid's bicubic
        filter, then run through the network on :attr:`device`.

        Args:
            images: Images in [-1, 1], shape ``[B, 3, H, W]``.

        Returns:
            Features of shape ``[B, 2048]`` in float64.
        """
        if self._net is None:
            self._net = _FIDInception().to(self.device)
        u8 = torch.round((images.clamp(-1, 1) + 1) * _HALF_RANGE).to(torch.uint8)
        resized = resize_batch(u8.permute(0, 2, 3, 1).cpu().numpy())
        x = (
            torch.from_numpy(resized).to(self.device).permute(0, 3, 1, 2)
            / _HALF_RANGE
            - 1.0
        )
        with (
            torch.no_grad(),
            torch.backends.cudnn.flags(enabled=True, allow_tf32=False),
        ):
            return _call(self._net, x).double().cpu().numpy()

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        """Add a batch of images.

        Args:
            images: Images in [-1, 1], shape ``[B, 3, H, W]``.
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

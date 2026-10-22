"""PyTorch metrics. Inputs: float images in [-1, 1], [B, 3, H, W]."""
import numpy as np
import torch
import torch.nn.functional as F

from .fid_core import feature_stats, frechet_distance, load_stats


def quantise(x: torch.Tensor) -> torch.Tensor:
    """Round to uint8 levels and return floats in [0, 1]."""
    return torch.round((x.clamp(-1, 1) + 1) * 127.5) / 255.0


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = ((quantise(x) - quantise(y)) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """11x11 Gaussian window, sigma 1.5, data range 1."""
    x, y = quantise(x), quantise(y)
    c = x.shape[1]
    g = torch.arange(11, dtype=torch.float32, device=x.device) - 5
    g = torch.exp(-(g ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(c, 1, -1, -1).contiguous()
    f = lambda a: F.conv2d(a, w, groups=c)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def lpips(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("LPIPS-VGG weights are not vendored yet (licence check pending).")


class FID:
    """Accumulator. `extractor` maps uint8 images [B,3,H,W] to 2048-d Inception features."""

    def __init__(self, extractor=None):
        self.extractor = extractor
        self._real_stats = None
        self.reset()

    @classmethod
    def from_stats(cls, path, extractor=None):
        self = cls(extractor)
        self._real_stats = load_stats(path)
        return self

    def reset(self):
        self._real, self._fake = [], []

    def update(self, images: torch.Tensor, real: bool):
        if self.extractor is None:
            raise RuntimeError("Inception weights are not vendored yet; pass an extractor.")
        u8 = torch.round((images.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu()
        (self._real if real else self._fake).append(self.extractor(u8).cpu().numpy())

    def compute(self) -> float:
        real = self._real_stats or feature_stats(np.concatenate(self._real))
        return frechet_distance(*real, *feature_stats(np.concatenate(self._fake)))

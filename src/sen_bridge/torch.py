"""PyTorch metrics: FID, LPIPS (VGG), PSNR, SSIM.

Inputs are float images in [-1, 1], shape [B, 3, H, W]. Every metric first rounds to uint8, so scores never
depend on the float precision of the outputs.

Evaluation only: rounding to uint8 has zero gradient, so never use these metrics as a training loss. Values outside
[-1, 1] are clipped silently (a model outputting 2.0 scores like one outputting 1.0), so check your output range. Rounding
is idempotent on 8-bit images (u / 127.5 - 1 -> same u), and 0.0 falls on a rounding tie and becomes ~+0.004; both are
inherent to the 8-bit encoding and identical in the two backends. Frozen settings (none are arguments):
FID: pytorch-fid Inception-v3 weights with clean-fid CPU bicubic resize to 299x299; LPIPS: VGG16 trunk, v0.1 linear
layers; SSIM: 11x11 Gaussian, sigma 1.5, data range 1.

The FID Inception block variants below are vendored from pytorch-fid (Apache-2.0, Copyright Maximilian Seitzer),
see NOTICE. LPIPS follows richzhang/PerceptualSimilarity (BSD-2-Clause).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision.models import vgg16
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE

from . import weights
from .fid_core import feature_stats, frechet_distance, load_stats, save_stats
from .resize import resize_batch


def quantise(x: torch.Tensor) -> torch.Tensor:
    """Round to uint8 levels; returns floats in [0, 1]."""
    return torch.round((x.clamp(-1, 1) + 1) * 127.5) / 255.0


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] -> [B], data range 1."""
    mse = ((quantise(x) - quantise(y)) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] -> [B]; valid-region mean over the three channels."""
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


# ---- LPIPS (VGG) ---------------------------------------------------------------------------------------------

class _LPIPS(nn.Module):
    SHIFT = (-0.030, -0.088, -0.188)
    SCALE = (0.458, 0.448, 0.450)
    TAPS = (3, 8, 15, 22, 29)  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3 in vgg16().features

    def __init__(self):
        super().__init__()
        self.vgg = vgg16(weights=None).features[:30]
        self.lins = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in (64, 128, 256, 512, 512)])
        sd = load_file(weights.path("lpips_vgg"))
        self.vgg.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vgg.")})
        for i, lin in enumerate(self.lins):
            lin.weight.data.copy_(sd[f"lin{i}.model.1.weight"])
        self.register_buffer("shift", torch.tensor(self.SHIFT).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor(self.SCALE).view(1, 3, 1, 1))
        self.eval().requires_grad_(False)

    def _feats(self, x):
        x, out = (x - self.shift) / self.scale, []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.TAPS:
                out.append(x)
        return out

    def forward(self, x, y):
        val = 0
        for fx, fy, lin in zip(self._feats(x), self._feats(y), self.lins):
            fx, fy = (f / (f.pow(2).sum(1, keepdim=True).sqrt() + 1e-10) for f in (fx, fy))
            val = val + lin((fx - fy) ** 2).mean((2, 3))
        return val.flatten()


_lpips_net = {}


def lpips(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] -> [B] LPIPS-VGG distance (lower = more similar)."""
    net = _lpips_net.get(x.device)
    if net is None:
        net = _lpips_net[x.device] = _LPIPS().to(x.device)
    q = lambda t: quantise(t) * 2 - 1
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):  # full precision on GPUs
        return net(q(x), q(y))


# ---- FID Inception (vendored variants, see module docstring) -------------------------------------------------

class _FIDInceptionA(InceptionA):
    def forward(self, x):  # TF's average pool excludes padding
        b = [self.branch1x1(x), self.branch5x5_2(self.branch5x5_1(x)),
             self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))]
        b.append(self.branch_pool(F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)))
        return torch.cat(b, 1)


class _FIDInceptionC(InceptionC):
    def forward(self, x):
        b7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        d = self.branch7x7dbl_5(self.branch7x7dbl_4(self.branch7x7dbl_3(self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))
        pool = self.branch_pool(F.avg_pool2d(x, 3, 1, 1, count_include_pad=False))
        return torch.cat([self.branch1x1(x), b7, d, pool], 1)


class _FIDInceptionE(InceptionE):
    def __init__(self, in_channels, pool="avg"):
        super().__init__(in_channels)
        self.pool = pool

    def forward(self, x):
        b3 = self.branch3x3_1(x)
        b3 = torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1)
        d = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        d = torch.cat([self.branch3x3dbl_3a(d), self.branch3x3dbl_3b(d)], 1)
        # FID's Inception uses max pooling in the last block (an artefact of the original TF graph)
        p = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False) if self.pool == "avg" else F.max_pool2d(x, 3, 1, 1)
        return torch.cat([self.branch1x1(x), b3, d, self.branch_pool(p)], 1)


class _FIDInception(nn.Module):
    def __init__(self):
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
        sd = {k: v for k, v in load_file(weights.path("fid_inception")).items() if not k.startswith("fc.")}
        self.load_state_dict(sd)
        self.eval().requires_grad_(False)

    def forward(self, x):
        """[B, 3, 299, 299] in [-1, 1] -> [B, 2048] pooled features."""
        x = self.Conv2d_3b_1x1(F.max_pool2d(self.Conv2d_2b_3x3(self.Conv2d_2a_3x3(self.Conv2d_1a_3x3(x))), 3, 2))
        x = F.max_pool2d(self.Conv2d_4a_3x3(x), 3, 2)
        for name in ("5b", "5c", "5d", "6a", "6b", "6c", "6d", "6e", "7a", "7b", "7c"):
            x = getattr(self, f"Mixed_{name}")(x)
        return x.mean((2, 3))


class FID:
    """Accumulator over real and generated images: update(images, real), compute(), reset()."""

    def __init__(self, device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._real_stats = None
        self.reset()

    @classmethod
    def from_stats(cls, path, device=None):
        self = cls(device)
        self._real_stats = load_stats(path)
        return self

    def reset(self):
        self._real, self._fake = [], []

    def features(self, images: torch.Tensor) -> np.ndarray:
        if self._net is None:
            self._net = _FIDInception().to(self.device)
        u8 = torch.round((images.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        resized = resize_batch(u8.permute(0, 2, 3, 1).cpu().numpy())  # CPU bicubic, float in [0, 255]
        x = torch.from_numpy(resized).to(self.device).permute(0, 3, 1, 2) / 127.5 - 1.0
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            return self._net(x).double().cpu().numpy()

    def update(self, images: torch.Tensor, real: bool):
        (self._real if real else self._fake).append(self.features(images))

    def stats(self, real: bool):
        return feature_stats(np.concatenate(self._real if real else self._fake))

    def save_stats(self, path, real: bool = True):
        save_stats(path, *self.stats(real))

    def compute(self) -> float:
        real = self._real_stats or self.stats(True)
        return frechet_distance(*real, *self.stats(False))

# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the PyTorch metrics against independent references."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from sen_bridge import fid_core, weights
from sen_bridge import torch as sbt
from sen_bridge.resize import resize_batch

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    Pair = tuple[NDArray[np.float32], NDArray[np.float32]]


def test_psnr_ssim_identity_and_ordering(pair: Pair) -> None:
    x = torch.from_numpy(pair[0])
    assert sbt.ssim(x, x).min() > 0.999
    assert sbt.psnr(x, x).min() > 90
    assert (sbt.psnr(x, x * 0.5) < sbt.psnr(x, x * 0.9)).all()


def test_psnr_ssim_match_skimage(pair: Pair) -> None:
    ski = pytest.importorskip("skimage.metrics")
    xt, yt = (torch.from_numpy(a) for a in pair)
    xs, ys = (sbt.quantise(t).permute(0, 2, 3, 1).numpy() for t in (xt, yt))
    psnr = [
        ski.peak_signal_noise_ratio(a, b, data_range=1)
        for a, b in zip(xs, ys, strict=True)
    ]
    ssim = [
        ski.structural_similarity(
            a,
            b,
            data_range=1,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        for a, b in zip(xs, ys, strict=True)
    ]
    assert np.abs(np.array(psnr) - sbt.psnr(xt, yt).numpy()).max() < 1e-4
    assert np.abs(np.array(ssim) - sbt.ssim(xt, yt).numpy()).max() < 1e-4


def test_frechet_zero_for_same_stats() -> None:
    feats = np.random.default_rng(0).normal(size=(500, 8))
    stats = fid_core.feature_stats(feats)
    assert fid_core.frechet_distance(*stats, *stats) == pytest.approx(0, abs=1e-6)


def test_stats_roundtrip(tmp_path: Path) -> None:
    mu, sigma = np.arange(3.0), np.eye(3)
    fid_core.save_stats(tmp_path / "s.npz", mu, sigma)
    got_mu, got_sigma = fid_core.load_stats(tmp_path / "s.npz")
    assert np.array_equal(got_mu, mu)
    assert np.array_equal(got_sigma, sigma)


def test_quantise_is_idempotent_on_8bit_images() -> None:
    levels = torch.arange(256, dtype=torch.uint8)
    decoded = levels.float() / 127.5 - 1  # the dataset's decoding
    back = torch.round(sbt.quantise(decoded) * 255).to(torch.uint8)
    assert torch.equal(back, levels)


def test_quantise_clips_and_blocks_gradients() -> None:
    assert float(sbt.quantise(torch.tensor(2.0))) == pytest.approx(1.0)
    assert float(sbt.quantise(torch.tensor(-2.0))) == pytest.approx(0.0)
    x = torch.tensor([0.3], requires_grad=True)
    sbt.quantise(x).sum().backward()
    assert x.grad is not None
    assert float(x.grad) == pytest.approx(0.0)


def test_scores_ignore_float_noise_below_half_a_step(pair: Pair) -> None:
    xt, yt = (torch.from_numpy(a) for a in pair)
    jitter = 1e-4 * torch.randn_like(xt)  # far below half an 8-bit step
    shifted = sbt.psnr(xt.clamp(-0.99, 0.99) + jitter, yt)
    assert torch.allclose(sbt.psnr(xt.clamp(-0.99, 0.99), yt), shifted, atol=0.05)


@pytest.mark.usefixtures("weights_ready")
def test_lpips_matches_reference_package(pair: Pair) -> None:
    ref = pytest.importorskip("lpips")
    xt, yt = (torch.from_numpy(a) for a in pair)

    def model_range(t: torch.Tensor) -> torch.Tensor:
        return sbt.quantise(t) * 2 - 1

    expected = ref.LPIPS(net="vgg", verbose=False)(model_range(xt), model_range(yt))
    assert (expected.flatten().detach() - sbt.lpips(xt, yt)).abs().max() < 1e-5


@pytest.mark.usefixtures("weights_ready")
def test_inception_features_match_pytorch_fid(pair: Pair) -> None:
    pf = pytest.importorskip("pytorch_fid.inception")
    xt = torch.from_numpy(pair[0][:4])
    u8 = torch.round((xt + 1) * 127.5).to(torch.uint8)
    resized = resize_batch(u8.permute(0, 2, 3, 1).numpy())
    unit = torch.from_numpy(resized).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        net = pf.InceptionV3([3], resize_input=False, normalize_input=True).eval()
        expected = net(unit)[0].flatten(1).double().numpy()
    got = sbt.FID(device="cpu").features(xt)
    assert np.abs(expected - got).max() < 1e-5


@pytest.mark.usefixtures("weights_ready")
def test_fid_accumulator_and_stats_roundtrip(pair: Pair, tmp_path: Path) -> None:
    xt, yt = (torch.from_numpy(a) for a in pair)
    fid = sbt.FID(device="cpu")
    fid.update(xt, real=True)
    fid.update(yt, real=False)
    value = fid.compute()
    assert value > 0
    fid.save_stats(tmp_path / "s.npz", real=True)
    loaded = sbt.FID.from_stats(tmp_path / "s.npz", device="cpu")
    loaded.update(yt, real=False)
    assert loaded.compute() == pytest.approx(value, rel=1e-6)
    fid.reset()


def test_weight_digests_are_pinned() -> None:
    assert all(len(src.sha256) == 64 for src in weights.SOURCES.values())

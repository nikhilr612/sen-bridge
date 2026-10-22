import numpy as np
import pytest

from sen_bridge import fid_core

torch = pytest.importorskip("torch")
from sen_bridge import torch as sbt  # noqa: E402


def test_psnr_ssim_identity_and_ordering(pair):
    x, _ = pair
    x = torch.from_numpy(x)
    assert sbt.ssim(x, x).min() > 0.999 and sbt.psnr(x, x).min() > 90
    assert (sbt.psnr(x, x * 0.5) < sbt.psnr(x, x * 0.9)).all()


def test_psnr_ssim_match_skimage(pair):
    ski = pytest.importorskip("skimage.metrics")
    x, y = pair
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    xs, ys = (sbt.quantise(t).permute(0, 2, 3, 1).numpy() for t in (xt, yt))
    ps = [ski.peak_signal_noise_ratio(a, b, data_range=1) for a, b in zip(xs, ys)]
    ss = [ski.structural_similarity(a, b, data_range=1, channel_axis=2, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False) for a, b in zip(xs, ys)]
    assert np.abs(ps - sbt.psnr(xt, yt).numpy()).max() < 1e-4
    assert np.abs(ss - sbt.ssim(xt, yt).numpy()).max() < 1e-4


def test_frechet_zero_for_same_stats():
    f = np.random.default_rng(0).normal(size=(500, 8))
    s = fid_core.feature_stats(f)
    assert fid_core.frechet_distance(*s, *s) == pytest.approx(0, abs=1e-6)


def test_quantisation_makes_scores_independent_of_float_noise(pair):
    x, y = pair
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    jitter = lambda t: t + 1e-4 * torch.randn_like(t)  # far below half a uint8 step (1/255)
    assert torch.allclose(sbt.psnr(xt, yt), sbt.psnr(jitter(xt.clamp(-0.99, 0.99)), yt), atol=0.05)


def test_lpips_matches_reference_package(pair, weights_ready):
    ref = pytest.importorskip("lpips")
    x, y = pair
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    q = lambda t: sbt.quantise(t) * 2 - 1
    r = ref.LPIPS(net="vgg", verbose=False)(q(xt), q(yt)).flatten().detach()
    assert (r - sbt.lpips(xt, yt)).abs().max() < 1e-5


def test_inception_features_match_pytorch_fid(pair, weights_ready):
    pf = pytest.importorskip("pytorch_fid.inception")
    from sen_bridge.resize import resize_batch
    x, _ = pair
    xt = torch.from_numpy(x[:4])
    u8 = torch.round((xt + 1) * 127.5).to(torch.uint8)
    rs = torch.from_numpy(resize_batch(u8.permute(0, 2, 3, 1).numpy())).permute(0, 3, 1, 2) / 255.0
    with torch.no_grad():
        ref = pf.InceptionV3([3], resize_input=False, normalize_input=True).eval()(rs)[0].flatten(1).double().numpy()
    assert np.abs(ref - sbt.FID(device="cpu").features(xt)).max() < 1e-5


def test_fid_accumulator_and_stats_roundtrip(pair, weights_ready, tmp_path):
    x, y = pair
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    f = sbt.FID(device="cpu")
    f.update(xt, True); f.update(yt, False)
    v = f.compute()
    assert v > 0
    f.save_stats(tmp_path / "s.npz", True)
    g = sbt.FID.from_stats(tmp_path / "s.npz", device="cpu")
    g.update(yt, False)
    assert g.compute() == pytest.approx(v, rel=1e-6)
    f.reset()
    assert not f._real and not f._fake


def test_weight_digests_are_pinned():
    from sen_bridge import weights
    assert all(s["sha256"] for s in weights.SOURCES.values())

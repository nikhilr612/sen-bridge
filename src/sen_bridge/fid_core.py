"""Framework-free FID maths and stats file format."""
import numpy as np
from scipy import linalg


def feature_stats(feats: np.ndarray):
    return feats.mean(0), np.cov(feats, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2, eps=1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def save_stats(path, mu, sigma):
    np.savez(path, mu=mu, sigma=sigma)


def load_stats(path):
    d = np.load(path)
    return d["mu"], d["sigma"]

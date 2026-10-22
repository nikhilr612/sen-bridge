import numpy as np
import pandas as pd
import pytest

from sen_bridge import filters, meta, preprocess as P, splits


def test_sar_roundtrip_and_shapes():
    db = np.random.default_rng(0).uniform(-25, -6, (2, 264, 264)).astype(np.float32)
    d = P.sar_to_db128(db)
    assert d.shape == (2, 128, 128)
    u = P.sar_to_uint8(d)
    assert u.shape == (3, 128, 128) and u.dtype == np.uint8
    assert np.abs(P.sar_from_uint8(u)[0] - d[0]).max() <= 25 / 255 / 2 + 1e-3


def test_multilook_is_linear_power():
    db = np.full((2, 264, 264), -10.0, np.float32)
    db[:, 4, 4] = 0.0  # crop starts at 4, so this is block (0, 0)
    assert P.sar_to_db128(db)[0, 0, 0] == pytest.approx(10 * np.log10((1 + 3 * 0.1) / 4), abs=1e-4)


def test_optical_roundtrip():
    rho = np.linspace(0, 0.3, 3 * 128 * 128, dtype=np.float32).reshape(3, 128, 128)
    assert np.abs(P.opt_from_uint8(P.opt_to_uint8(rho)) - rho).max() < 0.002
    assert P.to_model_range(np.array([0, 255], np.uint8)).tolist() == [-1.0, 1.0]


def test_meta_vector():
    m = meta.meta_vector(100, 45.0, 10.0, 24.0, True)
    assert m.shape == (7,) and m.dtype == np.float32
    assert m[5] == pytest.approx(0.5) and m[6] == 1
    assert np.linalg.norm(m[2:5]) == pytest.approx(1)


def test_metadata_filter():
    ok = filters.metadata_ok([10, 60, 10, 10], [50, 0, 50, -50], [7, 7, 1, 1])
    assert ok.tolist() == [True, False, False, True]


def test_splits_and_downsample():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"lat": rng.uniform(-60, 60, 2000), "lon": rng.uniform(-180, 180, 2000)})
    df["cluster_id"] = splits.assign_clusters(df, df.sample(50, random_state=0))
    df["stratum"] = np.where(df.lat > 0, "N", "S")
    df["split"] = splits.enforce_min_distance(df, splits.assign_splits(df))
    assert (df.groupby("cluster_id").split.nunique() <= 2).all()
    df["region"] = rng.choice(list("ABC"), len(df), p=[.6, .3, .1])
    assert len(splits.stratified_downsample(df, 500)) == 500

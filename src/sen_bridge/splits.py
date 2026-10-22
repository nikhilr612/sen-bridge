"""Geographic-cluster splits, 5 km leakage check, stratified downsampling."""
import numpy as np
import pandas as pd

from . import constants as C

_R = 6371.0


def _unit(lat, lon):
    p, l = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)], -1)


def assign_clusters(df: pd.DataFrame, seeds: pd.DataFrame) -> pd.Series:
    """Nearest seed city (lat, lon columns) per row."""
    from scipy.spatial import cKDTree
    _, idx = cKDTree(_unit(seeds.lat.values, seeds.lon.values)).query(_unit(df.lat.values, df.lon.values))
    return pd.Series(idx, index=df.index, name="cluster_id")


def assign_splits(df: pd.DataFrame, fractions=(0.835, 0.033, 0.132), seed=0) -> pd.Series:
    """Whole clusters -> split, stratified by `stratum` column (continent/hemisphere/climate).

    Fractions are of clusters per stratum; test is sized to reach >= 20k after filtering.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series("train", index=df.index, name="split")
    cl = df.drop_duplicates("cluster_id")[["cluster_id", "stratum"]]
    for _, g in cl.groupby("stratum"):
        ids = rng.permutation(g.cluster_id.values)
        n_val, n_test = int(round(len(ids) * fractions[1])), int(round(len(ids) * fractions[2]))
        out[df.cluster_id.isin(ids[:n_val])] = "val"
        out[df.cluster_id.isin(ids[n_val:n_val + n_test])] = "test"
    return out


def enforce_min_distance(df: pd.DataFrame, split: pd.Series, km=C.SPLIT_MIN_DIST_KM) -> pd.Series:
    """Move val/test rows within `km` of any train row to train."""
    from scipy.spatial import cKDTree
    split = split.copy()
    tr = split == "train"
    if tr.all() or not tr.any():
        return split
    tree = cKDTree(_unit(df.lat[tr].values, df.lon[tr].values))
    d, _ = tree.query(_unit(df.lat[~tr].values, df.lon[~tr].values))
    split[split.index[~tr][d * _R < km]] = "train"
    return split


def stratified_downsample(df: pd.DataFrame, n: int, region_col="region", pre_share: dict | None = None,
                          slack=0.03, seed=0) -> pd.DataFrame:
    """Sample n rows; cap each region at its pre-filter share + `slack`."""
    rng = np.random.default_rng(seed)
    if len(df) <= n:
        return df
    share = pre_share or (df[region_col].value_counts(normalize=True)).to_dict()
    caps = {r: int(np.floor((share.get(r, 0) + slack) * n)) for r in df[region_col].unique()}
    avail = df[region_col].value_counts()
    take = {r: min(avail[r], caps[r]) for r in avail.index}
    # redistribute leftover proportionally to regions with headroom
    while sum(take.values()) < n:
        room = {r: min(avail[r], caps[r]) - take[r] for r in take if take[r] < min(avail[r], caps[r])}
        if not room:
            caps = {r: avail[r] for r in avail.index}  # caps infeasible: relax
            continue
        take[max(room, key=room.get)] += 1
    while sum(take.values()) > n:
        take[max(take, key=take.get)] -= 1
    parts = [g.iloc[rng.permutation(len(g))[:take[r]]] for r, g in df.groupby(region_col)]
    return pd.concat(parts).sort_index()


def cap_share(df: pd.DataFrame, member: pd.Series, cap: float, seed=0) -> pd.DataFrame:
    """Within each split, keep rows where `member` is True at no more than `cap` of that split."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("split"):
        m = member.loc[g.index]
        inside, rest = g[m], g[~m]
        max_in = int(cap * len(rest) / (1 - cap))
        if len(inside) > max_in:
            inside = inside.iloc[rng.permutation(len(inside))[:max_in]]
        keep += [rest, inside]
    return pd.concat(keep).sort_index()


def cap_water(df: pd.DataFrame, cap=C.WATER_SPLIT_CAP, seed=0, col="water_frac_opt") -> pd.DataFrame:
    """>90% water crops, by the optical mask (radar VV misses wind-roughened sea), capped per split."""
    return cap_share(df, df[col] > C.WATER_CROP_FRAC, cap, seed)

"""Blind audit of the filters.

  python build/audit.py sample PILOT_DIR [--kept 200] [--per-reason 20] [--seed 0]
      -> PILOT_DIR/audit/sheet_XX.png (numbered pairs, SAR above optical, no decision shown)
         PILOT_DIR/audit/audit.csv (number, id, decision, scores; empty `label`)
  python build/audit.py score PILOT_DIR/audit/audit.csv
      -> per-decision rates from the filled `label` column (good | bad)

Labelling: look at each numbered pair and write `good` if radar and optical show the same ground
with a usable optical target, otherwise `bad` (and a short reason in `note`).
Sampling: --kept random kept pairs plus --per-reason pairs for each rejection reason.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

COLS, ROWS = 6, 4  # pairs per sheet = 24


def sample(d: Path, kept_n: int, per_reason: int, seed: int, exclude: Path | None = None):
    rng = np.random.default_rng(seed)
    kept = pd.read_parquet(d / "pilot.parquet")
    rej = pd.read_parquet(d / "rejects.parquet")
    rows = []
    k = kept.sample(min(kept_n, len(kept)), random_state=seed)
    rows += [(r.id, "kept", d / "sar" / f"{r.id}.png", d / "optical" / f"{r.id}.png", r) for r in k.itertuples()]
    if exclude:  # ids already audited, so a post-freeze draw is fresh
        seen = set(pd.read_csv(exclude).id)
        kept, rej = kept[~kept.id.isin(seen)], rej[~rej.id.isin(seen)]
        k = kept.sample(min(kept_n, len(kept)), random_state=seed)
        rows = [(r.id, "kept", d / "sar" / f"{r.id}.png", d / "optical" / f"{r.id}.png", r) for r in k.itertuples()]
    reasons = [r for r in rej.reason.unique() if (d / "rejects" / r).exists()]
    per = per_reason
    for reason in reasons:
        have = rej[(rej.reason == reason) & rej.id.map(lambda i: (d / "rejects" / reason / "sar" / f"{i}.png").exists())]
        for r in have.sample(min(per, len(have)), random_state=seed).itertuples():
            rows.append((r.id, reason, d / "rejects" / reason / "sar" / f"{r.id}.png",
                         d / "rejects" / reason / "optical" / f"{r.id}.png", r))
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order]
    out = d / "audit"
    out.mkdir(exist_ok=True)
    recs = []
    per_sheet = COLS * ROWS
    for s0 in range(0, len(rows), per_sheet):
        sheet = Image.new("RGB", (COLS * 132, ROWS * 272), (30, 30, 30))
        dr = ImageDraw.Draw(sheet)
        for j, (id_, dec, sp, op, r) in enumerate(rows[s0:s0 + per_sheet]):
            no = s0 + j + 1
            x, y = (j % COLS) * 132, (j // COLS) * 272
            sheet.paste(Image.open(sp), (x + 2, y + 14))
            sheet.paste(Image.open(op), (x + 2, y + 14 + 128))
            dr.text((x + 4, y + 2), f"#{no}", fill=(255, 255, 0))
            recs.append({"no": no, "sheet": s0 // per_sheet + 1, "id": id_, "decision": dec,
                         **{c: getattr(r, c, np.nan) for c in (
                             "cloud_rule_frac", "sensei_cloud_frac", "haze_score", "snow_score", "rfi_score",
                             "sat_frac", "water_frac_sar", "slope_std", "elev_mean", "dt_hours")},
                         "label": "", "note": ""})
        sheet.save(out / f"sheet_{s0 // per_sheet + 1:02d}.png")
    pd.DataFrame(recs).to_csv(out / "audit.csv", index=False)
    print(f"{len(recs)} pairs on {len(recs) // per_sheet + 1} sheets in {out}")


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def score(csv: Path):
    df = pd.read_csv(csv)
    df = df[df.label.isin(["good", "bad"])].copy()
    if df.empty:
        sys.exit("no labelled rows")
    df["note"] = df.note.fillna("")
    df["sure_bad"] = (df.label == "bad") & ~df.note.str.startswith("unsure")
    df["any_bad"] = df.label == "bad"
    kept = df[df.decision == "kept"]
    for col, name in (("sure_bad", "sure only"), ("any_bad", "incl. unsure")):
        k, n = int(kept[col].sum()), len(kept)
        lo, hi = _wilson(k, n)
        print(f"kept ({name}): {k}/{n} bad = {k / n:.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    print(kept[kept.any_bad].note.str.replace("unsure:", "").value_counts().to_string())
    for dec, g in df[df.decision != "kept"].groupby("decision"):
        print(f"{dec:>14}: n={len(g):3d}  truly bad {g.any_bad.mean():.0%} (sure {g.sure_bad.mean():.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"]); ap.add_argument("path")
    ap.add_argument("--kept", type=int, default=200); ap.add_argument("--per-reason", type=int, default=20); ap.add_argument("--exclude"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sample(Path(a.path), a.kept, a.per_reason, a.seed, Path(a.exclude) if a.exclude else None) if a.cmd == "sample" else score(Path(a.path))

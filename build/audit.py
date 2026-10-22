# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Blind audit of the filters.

Usage::

    python build/audit.py sample PILOT_DIR [--kept 200] [--per-reason 20]
        [--seed 0] [--exclude audited.csv]
    python build/audit.py score PILOT_DIR/audit/audit.csv

``sample`` writes ``PILOT_DIR/audit/sheet_XX.png`` (numbered pairs, radar above
optical, no decision shown) and ``audit.csv`` (number, id, decision, scores and an
empty ``label``). ``score`` reports per-decision rates from the filled ``label``
column.

Labelling: look at each numbered pair and write ``good`` if radar and optical show
the same ground with a usable optical target, otherwise ``bad`` (and a short reason
in ``note``; start it with ``unsure`` for low-confidence calls).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

_COLS, _ROWS = 6, 4  # pairs per sheet: 24
_TILE_W, _TILE_H, _PAD = 132, 272, 2
_IMG = 128
_LABEL_Y = 2
_IMAGE_Y = 14
_BACKGROUND = (30, 30, 30)
_YELLOW = (255, 255, 0)
_Z95 = 1.96
SCORE_COLUMNS = (
    "cloud_rule_frac",
    "sensei_cloud_frac",
    "haze_score",
    "snow_score",
    "rfi_score",
    "sat_frac",
    "water_frac_sar",
    "slope_std",
    "elev_mean",
    "dt_hours",
)


class Item(NamedTuple):
    """One pair drawn for the audit.

    Attributes:
        id: Example id.
        decision: ``"kept"`` or the rejection reason.
        sar: Path of the radar PNG.
        optical: Path of the optical PNG.
        scores: Recorded scores of the crop.
    """

    id: str
    decision: str
    sar: Path
    optical: Path
    scores: dict[str, float]


def _item(d: Path, row: dict[str, Any], decision: str) -> Item:
    """Build an item from a table row.

    Args:
        d: Pilot directory.
        row: Row of the kept or rejects table.
        decision: ``"kept"`` or the rejection reason.

    Returns:
        The item with the image paths for that decision.
    """
    base = d if decision == "kept" else d / "rejects" / decision
    sample_id = str(row["id"])
    scores = {c: float(row.get(c, np.nan)) for c in SCORE_COLUMNS}
    return Item(
        sample_id,
        decision,
        base / "sar" / f"{sample_id}.png",
        base / "optical" / f"{sample_id}.png",
        scores,
    )


def draw_items(
    d: Path,
    kept_n: int,
    per_reason: int,
    seed: int,
    exclude: Path | None = None,
) -> list[Item]:
    """Draw the audit sample.

    Args:
        d: Pilot directory with ``pilot.parquet``, ``rejects.parquet`` and images.
        kept_n: Number of kept pairs.
        per_reason: Number of pairs for each rejection reason.
        seed: Seed of the draw and of the display order.
        exclude: CSV with an ``id`` column of pairs to leave out (already audited).

    Returns:
        The items in random display order.
    """
    kept = pd.read_parquet(d / "pilot.parquet")
    rejects = pd.read_parquet(d / "rejects.parquet")
    if exclude:
        seen = set(pd.read_csv(exclude)["id"])
        kept, rejects = (
            kept[~kept["id"].isin(seen)],
            rejects[~rejects["id"].isin(seen)],
        )
    drawn = kept.sample(min(kept_n, len(kept)), random_state=seed)
    items = [_item(d, row, "kept") for row in drawn.to_dict("records")]
    for reason in rejects["reason"].unique():
        folder = d / "rejects" / str(reason) / "sar"
        have = rejects[rejects["reason"] == reason]
        have = have[have["id"].map(lambda i, f=folder: (f / f"{i}.png").exists())]
        pick = have.sample(min(per_reason, len(have)), random_state=seed)
        items += [_item(d, row, str(reason)) for row in pick.to_dict("records")]
    order = np.random.default_rng(seed).permutation(len(items))
    return [items[i] for i in order]


def make_sheets(items: list[Item], out: Path) -> pd.DataFrame:
    """Render numbered contact sheets and the label table.

    Args:
        items: Items in display order.
        out: Directory for the sheets.

    Returns:
        The label table: number, sheet, id, decision, scores and empty
        ``label`` and ``note`` columns.
    """
    per_sheet = _COLS * _ROWS
    records = []
    for start in range(0, len(items), per_sheet):
        sheet = Image.new("RGB", (_COLS * _TILE_W, _ROWS * _TILE_H), _BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        for j, item in enumerate(items[start : start + per_sheet]):
            number = start + j + 1
            x, y = (j % _COLS) * _TILE_W, (j // _COLS) * _TILE_H
            sheet.paste(Image.open(item.sar), (x + _PAD, y + _IMAGE_Y))
            sheet.paste(Image.open(item.optical), (x + _PAD, y + _IMAGE_Y + _IMG))
            draw.text((x + 2 * _PAD, y + _LABEL_Y), f"#{number}", fill=_YELLOW)
            records.append({
                "no": number,
                "sheet": start // per_sheet + 1,
                "id": item.id,
                "decision": item.decision,
                **item.scores,
                "label": "",
                "note": "",
            })
        sheet.save(out / f"sheet_{start // per_sheet + 1:02d}.png")
    return pd.DataFrame(records)


def wilson(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Compute a Wilson score interval for a proportion.

    Args:
        k: Number of successes.
        n: Number of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        The lower and upper bounds, or NaN twice if ``n`` is zero.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def score(csv: Path) -> None:
    """Print bad-pair rates from a filled label table.

    Args:
        csv: The ``audit.csv`` with the ``label`` column filled in.
    """
    df = pd.read_csv(csv)
    df = df[df["label"].isin(["good", "bad"])].copy()
    if df.empty:
        sys.exit("no labelled rows")
    df["note"] = df["note"].fillna("")
    df["any_bad"] = df["label"] == "bad"
    df["sure_bad"] = df["any_bad"] & ~df["note"].str.startswith("unsure")
    kept = df[df["decision"] == "kept"]
    for col, name in (("sure_bad", "sure only"), ("any_bad", "incl. unsure")):
        k, n = int(kept[col].sum()), len(kept)
        lo, hi = wilson(k, n)
        print(
            f"kept ({name}): {k}/{n} bad = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%})"
        )
    print(
        kept[kept["any_bad"]]["note"]
        .str.replace("unsure:", "")
        .value_counts()
        .to_string()
    )
    for decision, group in df[df["decision"] != "kept"].groupby("decision"):
        print(
            f"{decision!s:>14}: n={len(group):3d}  "
            f"truly bad {group['any_bad'].mean():.0%} "
            f"(sure {group['sure_bad'].mean():.0%})"
        )


def main() -> None:
    """Run the ``sample`` or ``score`` command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sample", "score"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--kept", type=int, default=200)
    parser.add_argument("--per-reason", type=int, default=20)
    parser.add_argument("--exclude", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "score":
        score(args.path)
        return
    items = draw_items(
        args.path, args.kept, args.per_reason, args.seed, args.exclude
    )
    out = args.path / "audit"
    out.mkdir(exist_ok=True)
    table = make_sheets(items, out)
    table.to_csv(out / "audit.csv", index=False)
    print(f"{len(table)} pairs on {int(table['sheet'].max())} sheets in {out}")


if __name__ == "__main__":
    main()

# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""Write ``README.md`` (Hugging Face dataset card and datasheet) into the release.

Usage::

    python build/card.py RELEASE_DIR

Reads ``RELEASE_DIR/build_stats.json`` and ``docs/thresholds.json``; everything else
is fixed text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from package import FEATURES, SPLITS

#: Filled in by the Hub when the dataset is published.
_DOI_PLACEHOLDER = "TODO"
_LARGE = 100_000
_ORDER = ("train", "val", "test")

_BODY = """
# SEN-Bridge

Paired Sentinel-1 radar and Sentinel-2 optical image crops (128x128, 20 m/pixel) for
benchmarking flow-matching and bridge models on radar-to-optical translation. Real
sensor coupling, no synthetic degradation.

- **Examples:** {n:,} ({train:,} train / {val:,} validation / {test:,} test).
- **Task:** sample the optical image given the radar image and metadata (`meta`:
  season, location, time gap, orbit).
- **Images:** 8-bit PNG. `sar` is R=VV, G=VH, B=VV-VH; `optical` is RGB
  (B4, B3, B2). Model input: `x / 127.5 - 1`.
- **Subsets by flag, not config:** `ds.filter(lambda r: r["clean"])`, or
  `abs(dt_hours) <= 24`.
- **Evaluation:** `scripts/evaluate.py` and the `sen-bridge` metrics (FID, LPIPS,
  PSNR, SSIM); real-target FID statistics are in `stats/`.

## Provenance

Derived from SSL4EO-S12 v1.1 (`embed2scale/SSL4EO-S12-v1.1`, Hugging Face commit
`{source_commit}`), CC BY 4.0. Only the train split of the source was used.
Build code commit: `{code_commit}` (see `scripts/build/`); seed {seed}.
Threshold snapshot SHA-256: `{sha256}`.

**Attribution.** Contains modified Copernicus Sentinel data 2019-2021. Please also
cite SSL4EO-S12 v1.1.

## Preprocessing (model-free, nothing stretched per image)

Centre-crop 264 to 256; 2x2 average to 128x128 (radar averaged in linear power,
then back to dB). Radar channels are clipped to VV [-25, 0] dB, VH [-32, -5] dB and
VV-VH [0, 15] dB and mapped to 0..255. Optical reflectance = DN/10000, clipped to
[0, 0.3], gamma 1/2.2, mapped to 0..255. Inverse mappings are in
`scripts/sen_bridge_build/preprocess.py`.

## Filters and yield

Candidates were location-seasons with |t_optical - t_radar| <= 48 h, outside local
winter at |lat| > 40, and SSL4EO cloud cover <= 5%. Of {fetched:,} fetched,
{kept:,} passed the image filters. Rejected IDs are in `rejects/rejects.parquet`.

| Rejected by | Count |
|---|---|
{rejects}

Radio-frequency interference and radar border noise are recorded (`rfi_score`) but
are not filters: the audit found them to reject almost only good pairs.

All constants (frozen in `docs/thresholds.json`):

| Constant | Value |
|---|---|
{constants}

## Splits

Whole 2x2-degree cells (a stand-in for SSL4EO's seed-city clusters, which are not
shipped) go to one split, stratified by region and hemisphere. Any validation or
test example within 5 km of a train example was moved to train. Crops that are over
90% water and crops that are at least 50% bare ground were capped at 10% and 8% of
each split. Region-stratified downsampling ran inside each split. Compute FID
against the test targets only, and select checkpoints on validation.

## Quality: what was checked

Three blind audit rounds on this pipeline's pilot (single reviewer, Claude; labels
in `docs/audit/`). After retuning, the clearly-bad rate among kept pairs was about
1-2% (n=120-150 per round, 95% CI up to about 6%); counting low-confidence calls it
was 7-10%, almost all thin haze or washed-out optical. The `clean` flag (neural
cloud mask exactly zero, textured optical or confirmed open water, mean luminance
0.10-0.80) covers {clean:.0%} of examples with about 2% low-confidence-bad and no
clearly bad pair in 297 audited. `clean` is biased toward vegetated and built
scenes. The estimated effect of all filters on the clearly-bad rate is about 2.0% to
1.1%.

## Known limitations

- Thin haze, small clouds (under 5% of a crop) and pale featureless land remain in
  some examples; no cheap rule separates them from valid scenes. Use `haze_score`,
  `flat_bright` and `clean`.
- Featureless bright arid crops were excluded, so deserts are under-represented
  relative to the source.
- Radar is not terrain-corrected: crops with slope std above 8 degrees are excluded
  (radar-optical shifts exceeded 3 px for 35-50% of crops above 6 degrees), but
  residual shifts of about 1 px (20 m) remain.
- Intertidal zones are not excluded by a tide model. A radar-versus-optical
  water-mismatch filter removes about 0.7% of crops. Radar water (VV < -18 dB)
  misses wind-roughened water; use `water_frac_opt`.
- `orbit` is derived from the local solar time of the radar pass (the source has no
  orbit field). `t_radar` and `t_optical` are assumed UTC.
- Pairs are near-simultaneous (48 h or less), not simultaneous; tides, wind and dew
  can differ. Near cities, 2019-2021, low relief outside the snow season.
- SSL4EO's neural cloud mask and land-cover layer were used for filtering and for
  the `lulc_*` flags only, never for pairing.
- LPIPS uses the VGG16 trunk, whose weights have no published licence; see
  `docs/LICENSES.md` before commercial use.

## Release policy

Fixes ship as new tagged revisions with a changelog; v1.0 is never overwritten. Test
targets are public; select models on validation only. Citation/DOI: {doi} (minted by
the Hub at publication).
"""


def front_matter(stats: dict[str, Any]) -> dict[str, Any]:
    """Build the YAML header of the dataset card.

    Args:
        stats: Contents of ``build_stats.json``.

    Returns:
        The header: licence, tags, split files and the typed features.
    """
    final = stats["final"]
    total = sum(final.values())
    splits = [
        {
            "name": SPLITS[k],
            "num_bytes": int(stats["split_bytes"][k]),
            "num_examples": int(final.get(k, 0)),
        }
        for k in _ORDER
    ]
    # The public API has no YAML export of the features.
    features = FEATURES._to_yaml_list()  # ruff: ignore[private-member-access]
    return {
        "license": "cc-by-4.0",
        "pretty_name": "SEN-Bridge",
        "task_categories": ["image-to-image"],
        "tags": [
            "remote-sensing",
            "sentinel-1",
            "sentinel-2",
            "sar",
            "flow-matching",
            "image-to-image",
        ],
        "size_categories": ["100K<n<1M"] if total >= _LARGE else ["10K<n<100K"],
        "configs": [
            {
                "config_name": "default",
                "data_files": [
                    {"split": SPLITS[k], "path": f"data/{SPLITS[k]}-*"}
                    for k in _ORDER
                ],
            }
        ],
        "dataset_info": {"features": features, "splits": splits},
    }


def body(stats: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Fill the card text with the build statistics and frozen thresholds.

    Args:
        stats: Contents of ``build_stats.json``.
        thresholds: Contents of ``docs/thresholds.json``.

    Returns:
        The Markdown body.
    """
    final = stats["final"]
    ranked = sorted(stats["rejects"].items(), key=lambda kv: -kv[1])
    return _BODY.format(
        n=sum(final.values()),
        train=final.get("train", 0),
        val=final.get("val", 0),
        test=final.get("test", 0),
        source_commit=thresholds["source_commit"],
        code_commit=thresholds["code_commit"],
        seed=thresholds["seed"],
        sha256=thresholds["sha256"],
        fetched=stats["candidates_fetched"],
        kept=stats["kept_by_filters"],
        rejects="\n".join(f"| {k} | {v:,} |" for k, v in ranked),
        constants="\n".join(
            f"| `{k}` | {v} |" for k, v in thresholds["constants"].items()
        ),
        clean=stats["clean_share"],
        doi=_DOI_PLACEHOLDER,
    )


def main() -> None:
    """Write the dataset card into the release directory given as argument."""
    release = Path(sys.argv[1])
    stats = json.loads((release / "build_stats.json").read_text(encoding="utf-8"))
    thresholds = json.loads(
        Path("docs/thresholds.json").read_text(encoding="utf-8")
    )
    header = yaml.safe_dump(front_matter(stats), sort_keys=False)
    (release / "README.md").write_text(
        f"---\n{header}---\n{body(stats, thresholds)}", encoding="utf-8"
    )
    print("wrote", release / "README.md")


if __name__ == "__main__":
    main()

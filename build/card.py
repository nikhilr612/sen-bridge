"""Write README.md (Hugging Face dataset card + datasheet) into the release directory.

Usage: python build/card.py RELEASE_DIR
Reads RELEASE_DIR/build_stats.json and docs/thresholds.json; everything else is fixed text.
"""
import json
import sys
from pathlib import Path

import yaml

from package import FEATURES, SPLITS

PLACEHOLDER = "TODO"


def main():
    rel = Path(sys.argv[1])
    st = json.loads((rel / "build_stats.json").read_text())
    th = json.loads(Path("docs/thresholds.json").read_text())
    final = st["final"]; n = sum(final.values())
    header = {
        "license": "cc-by-4.0", "pretty_name": "SEN-Bridge",
        "task_categories": ["image-to-image"],
        "tags": ["remote-sensing", "sentinel-1", "sentinel-2", "sar", "flow-matching", "image-to-image"],
        "size_categories": ["100K<n<1M"] if n >= 100_000 else ["10K<n<100K"],
        "configs": [{"config_name": "default", "data_files": [{"split": SPLITS[k], "path": f"data/{SPLITS[k]}-*"} for k in ("train", "val", "test")]}],
        "dataset_info": {"features": FEATURES._to_yaml_list(),
                         "splits": [{"name": SPLITS[k], "num_bytes": int(st["split_bytes"][k]), "num_examples": int(final.get(k, 0))} for k in ("train", "val", "test")]},
    }
    rej = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(st["rejects"].items(), key=lambda kv: -kv[1]))
    const = "\n".join(f"| `{k}` | {v} |" for k, v in th["constants"].items())
    body = f"""
# SEN-Bridge

Paired Sentinel-1 radar and Sentinel-2 optical image crops (128x128, 20 m/pixel) for benchmarking flow-matching and bridge
models on radar-to-optical translation. Real sensor coupling, no synthetic degradation.

- **Examples:** {n:,} ({final.get('train', 0):,} train / {final.get('val', 0):,} validation / {final.get('test', 0):,} test).
- **Task:** sample the optical image given the radar image and metadata (`meta`: season, location, time gap, orbit).
- **Images:** 8-bit PNG. `sar` is R=VV, G=VH, B=VV-VH; `optical` is RGB (B4, B3, B2). Model input: `x / 127.5 - 1`.
- **Subsets by flag, not config:** `ds.filter(lambda r: r["clean"])`, or `abs(dt_hours) <= 24`.

## Provenance
Derived from SSL4EO-S12 v1.1 (`embed2scale/SSL4EO-S12-v1.1`, Hugging Face commit `{th['source_commit']}`), CC BY 4.0.
Only the train split of the source was used. Build code commit: `{th['code_commit']}` (see `scripts/build/`); seed {th['seed']}.
Threshold snapshot SHA-256: `{th['sha256']}`.

**Attribution.** Contains modified Copernicus Sentinel data 2019-2021. Please also cite SSL4EO-S12 v1.1.

## Preprocessing (model-free, nothing stretched per image)
Centre-crop 264->256; 2x2 average to 128x128 (radar averaged in linear power, then back to dB).
Radar channels clipped to VV [-25, 0] dB, VH [-32, -5] dB, VV-VH [0, 15] dB and mapped to 0..255.
Optical reflectance = DN/10000, clipped to [0, 0.3], gamma 1/2.2, mapped to 0..255. Inverse mappings: `sen_bridge.preprocess`.

## Filters and yield
Candidates were location-seasons with |t_optical - t_radar| <= 48 h, outside local winter at |lat| > 40, and SSL4EO cloud cover <= 5%.
Of {st['candidates_fetched']:,} fetched, {st['kept_by_filters']:,} passed the image filters. Rejected IDs are in `rejects/rejects.parquet`.

| Rejected by | Count |
|---|---|
{rej}

All constants (frozen in `docs/thresholds.json`):

| Constant | Value |
|---|---|
{const}

## Splits
Whole 2x2-degree cells (a stand-in for SSL4EO's seed-city clusters, which are not shipped) go to one split, stratified by region and hemisphere;
moved to train any validation/test example within 5 km of a train example; capped crops that are >90% water and >=50% bare ground at 10% / 8% of each split;
region-stratified downsampling ran inside each split. FID must be computed against test targets only; select checkpoints on validation.

## Quality: what was checked
Three blind audit rounds on this pipeline's pilot (single reviewer, Claude; labels in the build repo). After retuning, the clearly-bad rate
among kept pairs was about 1-2% (n=120-150 per round, 95% CI up to ~6%); counting low-confidence calls it was 7-10%, almost all thin haze or
washed-out optical. The `clean` flag (neural cloud mask exactly zero, textured optical or confirmed open water, mean luminance 0.10-0.80)
covers {st['clean_share']:.0%} of examples with ~2% low-confidence-bad and no clearly bad pair in 297 audited. `clean` is biased toward
vegetated and built scenes. Estimated effect of all filters on the clearly-bad rate: ~2.0% -> ~1.1%.

## Known limitations
- Thin haze, small clouds (<5% of a crop) and pale featureless land remain in some examples; no cheap rule separates them from valid scenes. Use `haze_score`, `flat_bright`, `clean`.
- Featureless bright arid crops were excluded, so deserts are under-represented relative to the source.
- Radar is not terrain-corrected: crops with slope std > 8 deg are excluded (radar-optical shifts exceeded 3 px for 35-50% of crops above 6 deg), but residual shifts of ~1 px (20 m) remain.
- Intertidal zones are not excluded by a tide model. A radar-versus-optical water-mismatch filter removes about 0.7% of crops. Radar water (VV < -18 dB) misses wind-roughened water; use `water_frac_opt`.
- `orbit` is derived from local solar time of the radar pass (the source has no orbit field). `t_radar`/`t_optical` are assumed UTC.
- Pairs are near-simultaneous (<=48 h), not simultaneous; tides, wind and dew can differ. Near cities, 2019-2021, low relief outside the snow season.
- SSL4EO's neural cloud mask and land-cover layer were used for filtering and for the `lulc_*` flags only, never for pairing.
- Precomputed FID statistics and the reference evaluation script are not included yet (Inception/LPIPS weights not vendored; licences unconfirmed).

## Release policy
Fixes ship as new tagged revisions with a changelog; v1.0 is never overwritten. Test targets are public; select models on validation only.
Citation/DOI: {PLACEHOLDER} (minted by the Hub at publication).
"""
    (rel / "README.md").write_text("---\n" + yaml.safe_dump(header, sort_keys=False) + "---\n" + body)
    print("wrote", rel / "README.md")


if __name__ == "__main__":
    main()

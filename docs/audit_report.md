# Filter audit (pilot: 20 shards, ~10.7k candidates)

Three blind rounds. Contact sheets showed only a `#number`; decisions and scores were hidden.
**All labels are by one reviewer (Claude).** `round*_blank_for_reviewer.csv` are empty for a second labeller;
score with `python build/audit.py score <csv>`. "Unsure" = my own low-confidence calls, reported separately.

| Round | Thresholds | Kept sampled | Bad (sure) | Bad (incl. unsure) |
|---|---|---|---|---|
| 1 | first draft | 200 | 4.0% (CI 2.0-7.7) | 8.5% |
| 2 | retuned from round 1 | 150 | 0.7% (CI 0.1-3.7) | 6.7% |
| 3 | retuned from round 2 (kept as frozen) | 120 | 1.7% (CI 0.5-5.9) | 10.0% |

Round 3 is the cleanest estimate but still not independent of thresholds fitted on rounds 1-2.

## Reject precision, round 3 (share of audited rejects that really were bad pairs)
overexposed 100% (sure 12%) · too_dark 88% · saturated 80% · cloud_opaque 75% · cloud_shadow 25% · elevation 12% · rfi 0%.
elevation/steep are scope filters, not quality filters. rfi rejects only ~0.2% of crops at z>20 and is likely unnecessary.
Pooled over all rounds, cloud_shadow (neural mask) rejects were bad 47% of the time at >=5% cover and 69% at >=15%; no clean cutoff.

## Known residuals
- **Thin haze / washed-out optical** is most of the "unsure" kept set. No cheap statistic separates it from valid pale desert.
- **Featureless bright cream** (mean>0.9, std<0.025): a rule catching it flags 14 audited crops, 9 bad / 5 good sand. Not adopted; user decision.
- Small cloud/shadow patches below 5% remain in kept crops.
- Intertidal/coastline exclusion not implemented (needs a static layer; licence unconfirmed).
- Dark threshold 0.08 rests on two data points (bad <=0.064, dimmest good seen 0.094).

## Radar-optical coregistration (build/coreg_check.py, 400 textured land crops)
Cross-modal gradient correlation, 20 m px, random-pair control. 64% of crops gave a reliable peak (smooth/water/haze crops cannot be measured).
Flat to moderate terrain: median shift 1 px, 83% within 1 px, no systematic offset. Slope std above 6 deg: 35-50% of crops shift by more than 3 px
(radar not terrain-corrected). `SLOPE_STD_MAX` set to 8 deg (placeholder was 15). Steep bins hold only 14-26 crops each; cutoff is approximate.
Resolution limit: whole-pixel shifts only.

## Water, bare ground, tides (pilot4, 9,636 kept crops, LULC added)
- LULC ships with SSL4EO (10 classes). Bare ground >=50% of a crop: 2.9% of crops, so the 8% bare cap is a no-op guard. "Rangeland" (steppe/arid fringe) is 34% of crops.
- The featureless-bright-cream crops (mean>0.88, std<0.03) are 3% of kept crops and 98% are >=50% bare+rangeland, so aridity, not geography, drives them.
- Radar water test (VV<-18 dB) misses wind-roughened sea: >90%-water crops are 5.1% by radar but 10.4% by the S2 mask. The water cap now uses the S2 mask.
- Radar says water / optical says land: 6.8% of crops, but 506 of 653 are arid (dark sand), not tidal. Only 71 (0.7%) also have mapped water; they are excluded as `water_mismatch`
  (tide or flood). Tides cannot be separated from floods without a tide model; the filter is a cheap proxy.

## Lean filter set
Removed as filters (kept as recorded scores `rfi_score`, `border_frac`): RFI (0 of 26 audited rejects were bad) and radar border noise (mostly good dark/water pairs at 1%, no rejects at 10%).
Active exclusions: invalid/no-data, black, saturated (>1%), featureless near-white, featureless bright arid, opaque cloud (>=50%), neural cloud/shadow (>=5%),
water mismatch, slope >8 deg, elevation >2000 m. Estimated effect (rounds 2-3): clearly-bad pairs ~2.0% -> ~1.1%; with my uncertain calls ~10.9% -> ~8.1%.
A `clean` flag (neural mask exactly 0, textured optical or confirmed water, mean 0.10-0.80) covers 63% of kept crops with 2.0% uncertain-bad and 0/297 clearly bad; not yet implemented.

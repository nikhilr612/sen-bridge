# Licences and what is verified

Checked against primary sources (the projects' own LICENSE files, the TensorFlow archive, the Copernicus legal notice),
not against search summaries. Open items are marked **OPEN**.

## Code

| Component | Licence | Verified from | Used for |
|---|---|---|---|
| sen-bridge (this repo) | Apache-2.0, Copyright 2026 Nikhil Raghavendra (chosen: compatible with the adapted Apache-2.0 code) | `LICENSE` | - |
| pytorch-fid | Apache-2.0 | its LICENSE file and README ("licensed under the Apache License 2.0"; original TTUR code also Apache-2.0) | FID Inception graph, Frechet distance |
| clean-fid | MIT | its LICENSE file | CPU bicubic resize |
| lpips (PerceptualSimilarity) | BSD-2-Clause | its LICENSE file | LPIPS algorithm |
| torchvision | BSD-3-Clause | its LICENSE file | dependency only |

Apache-2.0 requires keeping the licence text, the NOTICE and a statement of changes; MIT and BSD-2 require keeping their notices.
All are in `NOTICE` and `licenses/`. The copyright holder is Nikhil Raghavendra, as named in
`NOTICE` and in every source file's header (enforced by the linter).

## Weights (converted copies hosted at `nikhilr01/sen-bridge-weights`)

| Weights | Source | Licence | Status |
|---|---|---|---|
| FID Inception-v3 | pytorch-fid release `fid_weights` (converted from the TensorFlow `inception-2015-12-05` archive) | The TF archive ships a `LICENSE`: Apache-2.0, (c) 2015 The TensorFlow Authors | Clear. SHA-256 pinned (its `6726825d` prefix matches the published file name). |
| LPIPS v0.1 VGG linear layers | lpips repository | BSD-2-Clause (repository licence) | Clear. **No published digest**: pinned trust-on-first-use. |
| VGG16 ImageNet trunk | torchvision `vgg16-397923af` | **OPEN.** No weights-specific licence is published. torchvision's docs say its pre-trained models may carry terms derived from the training data and that users must check. The Oxford VGG page says the *original* models are CC BY, but I did not confirm torchvision's weights are those. | SHA-256 pinned (prefix matches the file name). **Redistributed** (converted to safetensors) in the weights repo, by the maintainer's explicit decision and at the maintainer's responsibility; the repo card asserts no licence for it and offers removal on request. Every LPIPS paper uses these weights; for a commercial use, verify. |

## Data

| Item | Licence | Status |
|---|---|---|
| SSL4EO-S12 v1.1 | CC BY 4.0 | Confirmed in the dataset card front matter (`license: cc-by-4.0`) at the pinned commit. |
| SEN-Bridge (derived) | CC BY 4.0 | Same as source. Credit SSL4EO-S12 v1.1. |
| Copernicus Sentinel data | EU legal notice: free, full and open access, notice required | Adapted data must carry: **"Contains modified Copernicus Sentinel data [Year]"**. Verified in the European Commission legal notice text. The card uses it. |
| FID statistics files | mean and covariance of Inception features of the dataset images | Derived from CC BY data; contain no weights. Fine to ship. |

## Decisions

1. **Code licence and holder: decided.** Apache-2.0; the holder is Nikhil Raghavendra, who
   confirmed they hold the IP. If the work was done under an employer's or institution's IP
   policy, that policy governs and this should be re-checked before release.
2. **VGG16 weights for LPIPS: keep, decided.** The maintainer's view is that the weights are
   unlikely to be an issue in practice. For the record, the upstream position is unchanged:
   torchvision publishes no separate weights licence and tells users to check the terms of its
   pre-trained models. The converted weights are now hosted in `nikhilr01/sen-bridge-weights`
   (decided explicitly by the maintainer, who accepts responsibility if a rights holder
   objects; the file would then be removed). Users who need certainty (for example for
   commercial use) should verify the terms themselves. This file will be updated if upstream states a licence for these weights.
3. **Dataset licence:** CC BY 4.0, inherited from SSL4EO-S12 v1.1; not a free choice.

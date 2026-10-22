# Copyright 2026 Nikhil Raghavendra
# SPDX-License-Identifier: Apache-2.0
"""Frozen constants of the SEN-Bridge release (see the datasheet).

Every threshold below is written to ``docs/thresholds.json`` with a SHA-256 by
``build/freeze.py`` and must not be tuned against model scores.
"""

# ---- Geometry and radiometry -------------------------------------------------

#: Side of the centre crop of the 264x264 SSL4EO patch, in 10 m pixels.
CROP = 256
#: Side of an example after 2x2 averaging (20 m per pixel).
SIZE = 128
#: Radar clip ranges in dB as (low, high) for VV, VH and VV-VH.
SAR_CLIPS = ((-25.0, 0.0), (-32.0, -5.0), (0.0, 15.0))
#: Upper reflectance clip of the optical image.
OPT_MAX = 0.3
#: Gamma applied to clipped reflectance.
OPT_GAMMA = 1 / 2.2
#: Digital number to reflectance divisor of Sentinel-2 L2A.
S2_SCALE = 10000.0

# ---- Pair selection ----------------------------------------------------------

#: Largest allowed absolute gap between radar and optical acquisition, in hours.
MAX_DT_HOURS = 48.0
#: Largest allowed share of cloud or shadow pixels per crop, from SSL4EO's neural
#: mask. The audit found every crop below 6% fine and most at 9% or more bad.
CLOUD_MAX_FRAC = 0.05
#: Share of the spectral cloud rule above which a crop is rejected. The rule is
#: otherwise only a score because roofs and pale sand trip it.
CLOUD_RULE_REJECT = 0.5
#: Share of optical pixels with reflectance of at least 1.0 above which a crop is
#: rejected.
SAT_MAX_FRAC = 0.01
#: Share of near-white pixels (minimum channel above 0.85) above which a featureless
#: crop is rejected.
WHITE_MAX_FRAC = 0.40
#: Luminance standard deviation below which a crop counts as featureless. Bad whites
#: were at most 0.02, textured sand, snow and roofs at least 0.06.
WHITE_FLAT_STD = 0.03
#: Mean 8-bit luminance below which a crop is rejected. Bad crops were 0.03 to
#: 0.064, the dimmest good crop seen was 0.094.
DARK_MEAN_MIN = 0.08
#: Mean luminance above which a featureless crop on arid land is rejected: cloud and
#: haze cannot be told from pale sand.
FLAT_BRIGHT_MEAN = 0.88
#: Share of bare plus rangeland LULC above which land counts as arid.
ARID_LULC_MIN = 0.50
#: Radar water share minus optical water share above which a crop with mapped water
#: is rejected (tide or flood mismatch).
WATER_MISMATCH = 0.3
#: Minimum LULC water share for the mismatch rule, which excludes dark sand that
#: radar reads as water.
LULC_WATER_MIN = 0.05
#: Slope standard deviation above which a crop is rejected, in degrees. Radar is not
#: terrain-corrected and shifts exceeded 3 px for 35 to 50% of crops above 6
#: degrees.
SLOPE_STD_MAX = 8.0
#: Mean elevation above which a crop is rejected, in metres.
ELEV_MAX_M = 2000.0
#: Latitude beyond which local winter months are excluded, in degrees.
SNOW_LAT = 40.0

# ---- Thresholds of recorded scores (the scores are not filters) --------------

#: Radar VV level below which a pixel counts as border noise, in dB.
BORDER_DB = -35.0
#: Radar VV level below which a pixel counts as water, in dB.
WATER_DB = -18.0
#: Radar VH level below which a pixel is under the noise floor, in dB.
NOISE_FLOOR_DB = -28.0

# ---- Splits and caps ---------------------------------------------------------

#: Water share above which a crop counts as water-dominated.
WATER_CROP_FRAC = 0.90
#: Largest share of each split made of water-dominated crops.
WATER_SPLIT_CAP = 0.10
#: Largest share of each split made of bare-ground crops (currently a no-op: about
#: 3% of crops).
BARE_SPLIT_CAP = 0.08
#: Bare-ground share above which a crop counts as bare ground.
BARE_CROP_FRAC = 0.50
#: Minimum distance between any validation or test crop and any train crop, in
#: kilometres.
SPLIT_MIN_DIST_KM = 5.0

# ---- The ``clean`` flag (a subset, never a filter) ---------------------------

#: Minimum luminance standard deviation of a ``clean`` crop (textured optical).
CLEAN_MIN_STD = 0.05
#: Open water counts as ``clean`` if the optical mask says water above this
#: share ...
CLEAN_WATER_OPT = 0.9
#: ... and the land-cover layer agrees above this share.
CLEAN_WATER_LULC = 0.5
#: Mean luminance range of a ``clean`` crop (exclusive): neither dark nor
#: washed out.
CLEAN_MEAN_RANGE = (0.10, 0.80)

#: Edges, in degrees of absolute latitude, of the climate bands used to stratify the
#: splits (a Koppen-style stand-in: tropics, subtropics, two temperate bands, high
#: latitudes). The 1-degree clusters align with every edge, so none straddles a
#: band.
LAT_BAND_EDGES = (20, 30, 40, 50)

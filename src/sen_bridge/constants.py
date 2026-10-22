"""Frozen constants of the SEN-Bridge release (see the datasheet)."""

CROP = 256  # centre crop of the 264x264 SSL4EO patch
SIZE = 128  # after 2x2 averaging (20 m/pixel)

# radar clip ranges in dB: (lo, hi) for VV, VH, VV-VH
SAR_CLIPS = ((-25.0, 0.0), (-32.0, -5.0), (0.0, 15.0))

OPT_MAX = 0.3  # reflectance clip
OPT_GAMMA = 1 / 2.2
S2_SCALE = 10000.0

MAX_DT_HOURS = 48.0
CLOUD_MAX_FRAC = 0.01
BORDER_DB = -35.0
BORDER_MAX_FRAC = 0.01
WATER_DB = -18.0
WATER_CROP_FRAC = 0.90
WATER_SPLIT_CAP = 0.10
NOISE_FLOOR_DB = -28.0
SLOPE_STD_MAX = 15.0  # placeholder, to be fixed by the pilot audit before freeze
ELEV_MAX_M = 2000.0
SNOW_LAT = 40.0
SPLIT_MIN_DIST_KM = 5.0

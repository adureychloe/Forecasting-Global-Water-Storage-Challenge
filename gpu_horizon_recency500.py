"""Five-head horizon model with a recent-window final fit."""

import os

os.environ["HORIZON_PREFIX"] = "horizon_recency500_age0_1_2_3_6_7p"
os.environ["HORIZON_FIT_ROWS"] = "500000"

import gpu_horizon_variant5  # noqa: E402,F401

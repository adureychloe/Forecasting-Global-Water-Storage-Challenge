"""Five-head horizon model with a recent 1M-row final fit."""

import os

os.environ["HORIZON_PREFIX"] = "horizon_recency1m_age0_1_2_3_6_7p"
os.environ["HORIZON_FIT_ROWS"] = "1000000"

import gpu_horizon_variant5  # noqa: E402,F401

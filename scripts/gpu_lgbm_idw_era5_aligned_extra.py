"""Extra seeds 123/9001 of 2010+ LightGBM aligned Test walk."""

import gpu_lgbm_idw_era5_aligned as base

base.runner.SEEDS = (123, 9001)
base.runner.PREFIX = "lgbm_horizon_idw_era5_aligned_extra"
base.runner.LOG = base.runner.RESULTS / f"{base.runner.PREFIX}.jsonl"
if __name__ == "__main__":
    raise SystemExit(base.custom_main())

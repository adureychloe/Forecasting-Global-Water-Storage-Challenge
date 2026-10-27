"""Depthwise CatBoost age-routed horizon heads for heterogeneous blending."""

import os

import gpu_horizon_runner as runner
import numpy as np

runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (2027, 77)
runner.PREFIX = os.environ.get("HORIZON_DEPTHWISE_PREFIX", "horizon_depthwise")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], runner.BUCKETS[:4], default=runner.BUCKETS[4])


runner.bucket = bucket


def depthwise_model(seed, iterations=500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations, depth=8, learning_rate=0.04, l2_leaf_reg=10,
        random_strength=1, grow_policy="Depthwise", loss_function="RMSE",
        eval_metric="RMSE", task_type="GPU", devices="0", random_seed=seed,
        border_count=128, verbose=False, allow_writing_files=False,
    )


runner.make_model = depthwise_model
raise SystemExit(runner.main())

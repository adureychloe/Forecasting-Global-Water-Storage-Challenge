"""Capacity sweep for the official age-routed horizon heads."""

from __future__ import annotations

import os

import gpu_horizon_runner as runner
import numpy as np

runner.SEEDS = (2027, 77)
runner.PREFIX = os.environ.get("HORIZON_CAPACITY_PREFIX", "horizon_capacity")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"

max_leaves = int(os.environ.get("HORIZON_MAX_LEAVES", "95"))

runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")


def capacity_bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


runner.bucket = capacity_bucket


def capacity_model(seed: int, iterations: int = 500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_strength=1,
        grow_policy="Lossguide",
        max_leaves=max_leaves,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )


runner.make_model = capacity_model
raise SystemExit(runner.main())

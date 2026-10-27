"""Robust-loss probe for age-routed horizon heads."""

from __future__ import annotations

import os

import numpy as np
import gpu_horizon_runner as runner

runner.SEEDS = (2027, 77)
runner.PREFIX = os.environ.get("HORIZON_LOSS_PREFIX", "horizon_loss_huber1")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")

loss = os.environ.get("HORIZON_LOSS", "Huber:delta=1")


def loss_bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


runner.bucket = loss_bucket


def loss_model(seed: int, iterations: int = 500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_strength=1,
        grow_policy="Lossguide",
        max_leaves=63,
        loss_function=loss,
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )


runner.make_model = loss_model
raise SystemExit(runner.main())

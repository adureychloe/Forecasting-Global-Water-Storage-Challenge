"""Ordered CatBoost horizon heads with lighter L2 regularization."""

import numpy as np
import gpu_horizon_runner as runner

runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (77, 2027)
runner.PREFIX = "horizon_ordered_l2_5"
runner.LOG = runner.RESULTS / "horizon_ordered_l2_5.jsonl"


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], runner.BUCKETS[:4], default=runner.BUCKETS[4])


runner.bucket = bucket


def ordered_l2_5(seed, iterations=500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations, depth=8, learning_rate=0.04, l2_leaf_reg=5,
        random_strength=1, grow_policy="SymmetricTree", boosting_type="Ordered",
        loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
        random_seed=seed, border_count=128, verbose=False, allow_writing_files=False,
    )


runner.make_model = ordered_l2_5
raise SystemExit(runner.main())

"""Symmetric-tree, plain-boosting age-routed horizon diversity probe."""

import os

import numpy as np

import gpu_horizon_runner as runner

runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
seed_text = os.environ.get("HORIZON_PLAIN_SEEDS", "2027,77")
runner.SEEDS = tuple(int(x) for x in seed_text.split(",") if x.strip())
PLAIN_DEPTH = int(os.environ.get("HORIZON_PLAIN_DEPTH", "8"))
PLAIN_LR = float(os.environ.get("HORIZON_PLAIN_LR", "0.04"))
PLAIN_L2 = float(os.environ.get("HORIZON_PLAIN_L2", "10"))
PLAIN_BOOTSTRAP = os.environ.get("HORIZON_PLAIN_BOOTSTRAP", "")
runner.PREFIX = os.environ.get("HORIZON_PLAIN_PREFIX", "horizon_plain")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], runner.BUCKETS[:4], default=runner.BUCKETS[4])


runner.bucket = bucket


def plain_model(seed, iterations=500):
    from catboost import CatBoostRegressor

    params = dict(
        iterations=iterations, depth=PLAIN_DEPTH, learning_rate=PLAIN_LR, l2_leaf_reg=PLAIN_L2,
        random_strength=1, grow_policy="SymmetricTree", boosting_type="Plain",
        loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
        random_seed=seed, border_count=128, verbose=False, allow_writing_files=False,
    )
    if PLAIN_BOOTSTRAP == "Bernoulli":
        params.update(bootstrap_type="Bernoulli", subsample=0.8)
    return CatBoostRegressor(**params)


runner.make_model = plain_model
raise SystemExit(runner.main())

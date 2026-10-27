"""Enhanced Ordered heads with 18-month block-frozen target statistics."""

import enhanced_features
import gpu_horizon_runner as runner
import target_stats_blockfrozen18_features

runner.core.build_features = enhanced_features.build_features
runner.augment_supervised_features = target_stats_blockfrozen18_features.augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.PREFIX = "catboost_horizon_enhanced_ordered_targetstats_blockfrozen18"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (77, 2027)


def bucket(age):
    import numpy as np

    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


runner.bucket = bucket


def model(seed, iterations=500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=5,
        random_strength=1,
        grow_policy="SymmetricTree",
        boosting_type="Ordered",
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )


runner.make_model = model
raise SystemExit(runner.main())

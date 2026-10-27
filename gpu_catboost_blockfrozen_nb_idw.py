"""Ordered CatBoost + block-frozen memory + IDW/250km/climate neighbourhood TWS."""

import enhanced_features
import gpu_horizon_runner as runner
import neighborhood_tws_features
import target_stats_blockfrozen_features
from coordinate_compliance import drop_coordinate_features

MASKED_JUNK_PREFIX = "TWS_t_masked"


def build_features(frame):
    features = enhanced_features.build_features(frame)
    features = neighborhood_tws_features.add_rich_neighborhood_features(frame, features)
    junk = [column for column in features.columns if str(column).startswith(MASKED_JUNK_PREFIX)]
    features = features.drop(columns=junk, errors="ignore")
    return drop_coordinate_features(features)


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_blockfrozen_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    return drop_coordinate_features(x_fit), drop_coordinate_features(x_query)


def bucket(age):
    import numpy as np

    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


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


runner.core.build_features = build_features
runner.augment_supervised_features = augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.PREFIX = "catboost_horizon_blockfrozen_nb_idw"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (2027, 77)
runner.bucket = bucket
runner.make_model = model
raise SystemExit(runner.main())

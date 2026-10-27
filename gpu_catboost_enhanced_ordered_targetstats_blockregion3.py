"""Block-frozen target history with coarse three-region Ordered heads."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import enhanced_features
import gpu_horizon_runner as runner
import target_stats_blockfrozen_features


AGE_BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
N_REGIONS = 3
runner.core.build_features = enhanced_features.build_features
runner.augment_supervised_features = target_stats_blockfrozen_features.augment_features
runner.BUCKETS = tuple(f"{age}_region{region}" for age in AGE_BUCKETS for region in range(N_REGIONS))
runner.PREFIX = "catboost_horizon_enhanced_ordered_targetstats_blockregion3"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (2027,)


def age_bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        AGE_BUCKETS[:4],
        default=AGE_BUCKETS[4],
    )


def spherical(raw):
    lat = np.deg2rad(raw["lat"].to_numpy(dtype=float))
    lon = np.deg2rad(raw["lon"].to_numpy(dtype=float))
    return np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )


def route_groups(x_fit, x_query, fit_raw, query_raw):
    coordinates = pd.concat(
        [fit_raw[["lat", "lon"]], query_raw[["lat", "lon"]]], ignore_index=True
    ).drop_duplicates()
    clusterer = KMeans(n_clusters=N_REGIONS, random_state=42, n_init=10).fit(
        spherical(coordinates)
    )
    fit_region = clusterer.predict(spherical(fit_raw))
    query_region = clusterer.predict(spherical(query_raw))
    fit_age = age_bucket(x_fit["TWS_t_age"].to_numpy(dtype=np.float32))
    query_age = age_bucket(x_query["TWS_t_age"].to_numpy(dtype=np.float32))
    return (
        np.asarray([f"{age}_region{region}" for age, region in zip(fit_age, fit_region)]),
        np.asarray([f"{age}_region{region}" for age, region in zip(query_age, query_region)]),
    )


runner.route_groups = route_groups


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

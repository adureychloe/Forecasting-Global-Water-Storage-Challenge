"""XGBoost CUDA heads with block-frozen memory and 500 km neighbourhood TWS."""

import enhanced_features
import gpu_xgb_horizon_runner as runner
import neighborhood_tws_features
import target_stats_blockfrozen_features
from coordinate_compliance import drop_coordinate_features

MASKED_JUNK_PREFIX = "TWS_t_masked"


def build_features(frame):
    features = enhanced_features.build_features(frame)
    features = neighborhood_tws_features.add_neighborhood_features(frame, features)
    junk = [column for column in features.columns if str(column).startswith(MASKED_JUNK_PREFIX)]
    features = features.drop(columns=junk, errors="ignore")
    return drop_coordinate_features(features)


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_blockfrozen_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    return drop_coordinate_features(x_fit), drop_coordinate_features(x_query)


def model(seed, n_estimators=900):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.045,
        min_child_weight=8.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=12.0,
        reg_alpha=0.10,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        max_bin=512,
        random_state=seed,
        n_jobs=16,
        early_stopping_rounds=90,
    )


runner.core.build_features = build_features
runner.augment_supervised_features = augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (27182, 2027)
runner.PREFIX = "xgb_horizon_blockfrozen_neighborhood"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.make_model = model
raise SystemExit(runner.main())

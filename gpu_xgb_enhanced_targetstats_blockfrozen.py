"""XGBoost CUDA horizon heads with strict block-frozen target memory."""

import enhanced_features
import gpu_xgb_horizon_runner as runner
import target_stats_blockfrozen_features

runner.core.build_features = enhanced_features.build_features
runner.augment_supervised_features = target_stats_blockfrozen_features.augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (27182,)
runner.PREFIX = "xgb_horizon_enhanced_targetstats_blockfrozen"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


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


runner.make_model = model
raise SystemExit(runner.main())

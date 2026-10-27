"""Enhanced-feature XGBoost CUDA depth-8 horizon heads."""

import enhanced_features
import gpu_xgb_horizon_runner as runner

runner.core.build_features = enhanced_features.build_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (42, 9001)
runner.PREFIX = "xgb_horizon_enhanced_depth8"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def model(seed, n_estimators=900):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.035,
        min_child_weight=10.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=16.0,
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

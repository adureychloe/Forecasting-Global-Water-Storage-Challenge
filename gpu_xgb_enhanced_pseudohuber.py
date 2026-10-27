"""Enhanced-feature XGBoost GPU horizon heads with pseudo-Huber loss."""

import enhanced_features
import gpu_xgb_horizon_runner as runner

runner.core.build_features = enhanced_features.build_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (31415, 27182)
runner.PREFIX = "xgb_horizon_enhanced_pseudohuber"
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
        objective="reg:pseudohubererror",
        huber_slope=1.0,
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

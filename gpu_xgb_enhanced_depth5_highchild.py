"""Enhanced-feature XGBoost CUDA depth5 heads with stronger child regularization."""

import enhanced_features
import gpu_xgb_horizon_runner as runner

runner.core.build_features = enhanced_features.build_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (27182, 2027)
runner.PREFIX = "xgb_horizon_enhanced_depth5_highchild"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def model(seed, n_estimators=900):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=20.0,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=20.0,
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

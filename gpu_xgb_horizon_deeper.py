"""Deeper/low-learning-rate XGBoost horizon heads for a second heterogeneous probe."""

import gpu_xgb_horizon_runner as runner


runner.PREFIX = "xgb_horizon5_deep"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (77, 123)


def deep_model(seed, n_estimators=1100):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=n_estimators, max_depth=10, learning_rate=0.022,
        min_child_weight=20.0, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=30.0, reg_alpha=0.1, objective="reg:squarederror",
        eval_metric="rmse", tree_method="hist", device="cuda", max_bin=256,
        random_state=seed, n_jobs=16, early_stopping_rounds=100,
    )


runner.make_model = deep_model
raise SystemExit(runner.main())

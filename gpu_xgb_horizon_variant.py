"""Moderate-capacity XGBoost GPU horizon heads for heterogeneity."""

import gpu_xgb_horizon_runner as runner

runner.PREFIX = "xgb_horizon5_variant"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (42, 9001)


def variant_model(seed, n_estimators=900):
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


runner.make_model = variant_model
raise SystemExit(runner.main())

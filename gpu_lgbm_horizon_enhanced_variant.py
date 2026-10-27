"""Second LightGBM parameterization on the exact public-feature horizon route."""

import enhanced_features
import gpu_lgbm_horizon as runner

runner.core.build_features = enhanced_features.build_features
runner.runner.core.build_features = enhanced_features.build_features
runner.runner.PREFIX = "lgbm_horizon_enhanced_variant"
runner.runner.LOG = runner.runner.RESULTS / "lgbm_horizon_enhanced_variant.jsonl"
runner.runner.SEEDS = (42, 9001)


def variant_model(seed, n_estimators=850):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=10.0,
        reg_alpha=0.05,
        objective="regression",
        metric="rmse",
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
        max_bin=127,
        random_state=seed,
        n_jobs=16,
        verbosity=-1,
    )


runner.make_model = variant_model
runner.runner.make_model = variant_model
raise SystemExit(runner.main())

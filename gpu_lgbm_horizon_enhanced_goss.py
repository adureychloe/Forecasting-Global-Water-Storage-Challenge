"""Enhanced-feature LightGBM GOSS horizon heads."""

import enhanced_features
import gpu_lgbm_horizon as base
import lightgbm as lgb

# LightGBM 4.x removed fit(verbose=..., early_stopping_rounds=...); the shared
# runner still uses that older signature, so translate it to callbacks locally.
_fit = lgb.LGBMRegressor.fit


def _fit_compat(self, *args, **kwargs):
    rounds = kwargs.pop("early_stopping_rounds", None)
    kwargs.pop("verbose", None)
    if rounds:
        kwargs["callbacks"] = [lgb.early_stopping(rounds, verbose=False)]
    return _fit(self, *args, **kwargs)


lgb.LGBMRegressor.fit = _fit_compat
lgb.LGBMRegressor.get_best_iteration = lambda self: int(self.best_iteration_ or self.n_estimators)

base.runner.core.build_features = enhanced_features.build_features
base.runner.PREFIX = "lgbm_horizon_enhanced_goss"
base.runner.LOG = base.runner.RESULTS / "lgbm_horizon_enhanced_goss.jsonl"
base.runner.SEEDS = (77, 2027)


def model(seed, n_estimators=850):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        boosting_type="goss",
        top_rate=0.20,
        other_rate=0.10,
        reg_lambda=15.0,
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


base.runner.make_model = model
raise SystemExit(base.runner.main())

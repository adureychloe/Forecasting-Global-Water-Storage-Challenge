"""2010+ XGBoost CUDA full-fit with time-aligned recursive Test inference.

CatBoost aligned transferred to 0.7105. LGBM aligned CSVs are ready. This
applies the same walk to XGB, the remaining tree family in the stack.
Skip old-protocol CV. Mix 2.5% after CSVs. Do not agent-submit.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

import enhanced_features
import era5_anomaly_features
import era5_cell_features
import gpu_horizon_runner as runner
import neighborhood_tws_features
import target_stats_blockfrozen_features
from aligned_feedback import walk_aligned
from coordinate_compliance import drop_coordinate_features
from gpu_dualhead_runner import mask_by_template, public_states

MASKED_JUNK_PREFIX = "TWS_t_masked"
INNER_ITERS = 1
FIT_START = pd.Timestamp("2010-01-01")


def build_features(frame):
    features = enhanced_features.build_features(frame)
    features = neighborhood_tws_features.add_rich_neighborhood_features(
        frame, features, fill_missing_with_anchor=True
    )
    features = era5_cell_features.add_era5_features(frame, features)
    junk = [column for column in features.columns if str(column).startswith(MASKED_JUNK_PREFIX)]
    features = features.drop(columns=junk, errors="ignore")
    return drop_coordinate_features(features)


def augment_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    x_fit, x_query = target_stats_blockfrozen_features.augment_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )
    x_fit, x_query = era5_anomaly_features.augment_features(x_fit, x_query, fit_raw, query_raw)
    return drop_coordinate_features(x_fit), drop_coordinate_features(x_query)


def bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


def model(seed, iterations=500, early_stopping_rounds=None):
    from xgboost import XGBRegressor

    params = {
        "n_estimators": iterations,
        "max_depth": 8,
        "learning_rate": 0.04,
        "min_child_weight": 8.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 12.0,
        "reg_alpha": 0.10,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "max_bin": 256,
        "random_state": seed,
        "n_jobs": 8,
        "verbosity": 0,
    }
    if early_stopping_rounds:
        params["early_stopping_rounds"] = early_stopping_rounds
    return XGBRegressor(**params)


def _predict_buckets(models, fallback, x_query, bquery):
    pred = np.zeros(len(x_query), dtype=np.float32)
    filled = np.zeros(len(x_query), dtype=bool)
    for name, head in models.items():
        idx = bquery == name
        if not idx.any() or head is None:
            continue
        transformed = head.predict(x_query.loc[idx]).astype(np.float32)
        pred[idx] = runner.inverse_predictions(transformed, x_query.loc[idx])
        filled[idx] = True
    if (~filled).any():
        pred[~filled] = fallback[~filled]
    return pred


def sequential_test_predict(models, train, train_input, test, y_train):
    n_train = len(train_input)

    def predict_month(history, current):
        combined = pd.concat([history, current], ignore_index=True, sort=False)
        features = (
            runner.core.build_features(combined)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
        )
        x_train, x_month = runner.augment_supervised_features(
            features.iloc[:n_train].copy(),
            y_train,
            features.iloc[-len(current) :].copy(),
            train,
            current,
        )
        _, bmonth = runner.route_groups(x_train, x_month, train, current)
        if "tws_neighbor_anchor_mean" in x_month.columns:
            fallback = x_month["tws_neighbor_anchor_mean"].to_numpy(dtype=np.float32)
        elif "TWS_t_ffill" in x_month.columns:
            fallback = x_month["TWS_t_ffill"].to_numpy(dtype=np.float32)
        else:
            fallback = np.zeros(len(x_month), dtype=np.float32)
        pred_m = _predict_buckets(models, fallback, x_month, bmonth)
        gc.collect()
        return pred_m

    def on_month(current, values):
        runner.log(
            {
                "stage": "aligned_month",
                "month": str(pd.Timestamp(current.time.iloc[0]).date()),
                "n": int(len(current)),
            }
        )

    return walk_aligned(train_input, test, predict_month, on_month)


def custom_main():
    train = pd.read_csv(runner.core.DATA / "Train.csv")
    test = pd.read_csv(runner.core.DATA / "Test.csv")
    sample = pd.read_csv(runner.core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    y_train = train.target.to_numpy(np.float32)
    recency = train.time >= FIT_START
    runner.log(
        {
            "stage": "start",
            "skip_cv": True,
            "reason": "CatBoost aligned transferred; apply same walk to XGB",
            "states": states.astype(int).tolist(),
            "buckets": list(runner.BUCKETS),
            "mode": "xgb_aligned_next_month",
            "fit_start": str(FIT_START.date()),
            "n_recency": int(recency.sum()),
            "inner_iters": INNER_ITERS,
        }
    )
    for seed in runner.SEEDS:
        train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
        test_input = test.copy()
        test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
        combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
        features = runner.core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        x_train, x_test = runner.augment_supervised_features(
            features.iloc[: len(train)].copy(),
            y_train,
            features.iloc[len(train) :].copy(),
            train,
            test,
        )
        bfit, btest = runner.route_groups(x_train, x_test, train, test)
        fit_selector = recency.to_numpy()
        models = {}
        static_pred = np.zeros(len(test), dtype=np.float32)
        for name in runner.BUCKETS:
            fit_idx = (bfit == name) & fit_selector
            test_idx = btest == name
            if fit_idx.sum() < 1000 or test_idx.sum() == 0:
                models[name] = None
                runner.log({"stage": "skip_full", "seed": seed, "bucket": name, "n_fit": int(fit_idx.sum())})
                continue
            runner.log({"stage": "fit_head", "seed": seed, "bucket": name, "n_fit": int(fit_idx.sum())})
            head = runner.make_model(seed, 500)
            head.fit(x_train.loc[fit_idx], runner.transform_targets(y_train[fit_idx], x_train.loc[fit_idx]), verbose=False)
            transformed = head.predict(x_test.loc[test_idx]).astype(np.float32)
            static_pred[test_idx] = runner.inverse_predictions(transformed, x_test.loc[test_idx])
            models[name] = head
        aligned_pred = sequential_test_predict(models, train, train_input, test, y_train)
        for label, pred in (("static", static_pred), ("aligned", aligned_pred)):
            if not np.isfinite(pred).all():
                raise ValueError(f"non-finite {label} predictions for seed {seed}")
            runner.core.make_submission_frame(test, pred, sample).to_csv(
                runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_{label}_raw.csv",
                index=False,
            )
        rms = float(np.sqrt(np.mean((aligned_pred - static_pred) ** 2)))
        runner.log({"stage": "full", "seed": seed, "aligned_vs_static_rms": rms})
        for name, head in list(models.items()):
            del head
        gc.collect()
    runner.log({"stage": "complete"})
    while True:
        import time as time_mod

        time_mod.sleep(60)
    return 0


runner.core.build_features = build_features
runner.augment_supervised_features = augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.PREFIX = "xgb_horizon_idw_era5_aligned"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (2027, 77)
runner.bucket = bucket
runner.make_model = model
if __name__ == "__main__":
    raise SystemExit(custom_main())

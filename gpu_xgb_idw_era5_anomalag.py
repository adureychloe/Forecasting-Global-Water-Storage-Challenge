"""XGBoost CUDA on the CatBoost #253 lagged-ERA5-anomaly stack.

Same IDW+ERA5+spatialiter+2010 recipe as #241, plus fit-only clim anomalies
at lags 1/2/3. lat/lon remain lookup keys only.
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
from coordinate_compliance import drop_coordinate_features
from gpu_dualhead_runner import mask_by_template, public_states

MASKED_JUNK_PREFIX = "TWS_t_masked"
INNER_ITERS = 3
FIT_START = pd.Timestamp("2010-01-01")
ANCHORS = (pd.Period("2007-09", freq="M"), pd.Period("2009-01", freq="M"))


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
    x_fit, x_query = era5_anomaly_features.augment_features(
        x_fit, x_query, fit_raw, query_raw, lags=(1, 2, 3)
    )
    return drop_coordinate_features(x_fit), drop_coordinate_features(x_query)


def bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


def model(seed, iterations=700, early_stopping_rounds=None):
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


def fit_head(head, x, y, eval_x=None, eval_y=None):
    if eval_x is None:
        head.fit(x, y, verbose=False)
        return
    head.fit(x, y, eval_set=[(eval_x, eval_y)], verbose=False)


def best_iters(head):
    best = getattr(head, "best_iteration", None)
    if best is None or int(best) < 0:
        return int(head.get_params().get("n_estimators", 0))
    return int(best) + 1


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
    test_work = test.copy().reset_index(drop=True)
    test_work.loc[test_work.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
    pred_all = np.zeros(len(test_work), dtype=np.float32)
    history_test = []
    n_train = len(train_input)
    for month, group in test_work.groupby("time", sort=True):
        pos = group.index.to_numpy()
        month_raw = test_work.loc[pos].copy()
        combined = pd.concat([train_input, *history_test, month_raw], ignore_index=True, sort=False)
        features = (
            runner.core.build_features(combined)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
        )
        x_train, x_month = runner.augment_supervised_features(
            features.iloc[:n_train].copy(),
            y_train,
            features.iloc[-len(month_raw) :].copy(),
            train,
            month_raw,
        )
        _, bmonth = runner.route_groups(x_train, x_month, train, month_raw)
        if "tws_neighbor_anchor_mean" in x_month.columns:
            fallback = x_month["tws_neighbor_anchor_mean"].to_numpy(dtype=np.float32)
        elif "TWS_t_ffill" in x_month.columns:
            fallback = x_month["TWS_t_ffill"].to_numpy(dtype=np.float32)
        else:
            fallback = np.zeros(len(x_month), dtype=np.float32)
        pred_m = _predict_buckets(models, fallback, x_month, bmonth)
        for _ in range(1, INNER_ITERS):
            x_month = neighborhood_tws_features.overlay_pred_neighborhood(month_raw, x_month, pred_m)
            x_month = drop_coordinate_features(x_month)
            pred_m = _predict_buckets(models, fallback, x_month, bmonth)
        pred_all[pos] = pred_m
        filled = month_raw.copy()
        masked = filled["TWS_t_masked"].to_numpy(dtype=bool)
        tws = pd.to_numeric(filled["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
        filled["TWS_t"] = np.where(masked, pred_m, tws)
        history_test.append(filled)
        runner.log({"stage": "xgb_anomalag_month", "month": str(pd.Timestamp(month).date()), "n": int(len(month_raw))})
        gc.collect()
    return pred_all


def pattern_fold_data(train, states, anchor):
    test_months = pd.PeriodIndex([anchor + i for i in range(len(states))], freq="M")
    all_months = train.time.dt.to_period("M")
    valid = train.loc[all_months.isin(test_months)].copy()
    fit = train.loc[all_months < anchor].copy()
    valid_month = valid.time.dt.to_period("M")
    state_map = dict(zip(test_months, states))
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    combined = mask_by_template(combined, states, align_validation=False)
    n = len(fit)
    exact = valid_month.map(state_map).to_numpy(dtype=bool)
    combined.loc[n:, "TWS_t_masked"] = exact
    combined.loc[n + np.flatnonzero(exact), "TWS_t"] = np.nan
    features = runner.core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y_fit = fit.target.to_numpy(np.float32)
    x_fit, x_valid = runner.augment_supervised_features(
        features.iloc[:n].copy(), y_fit, features.iloc[n:].copy(), fit, valid
    )
    bfit, bvalid = runner.route_groups(x_fit, x_valid, fit, valid)
    return x_fit, y_fit, x_valid, valid.target.to_numpy(np.float32), bfit, bvalid, exact, str(test_months[0]), str(test_months[-1])


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
            "states": states.astype(int).tolist(),
            "buckets": list(runner.BUCKETS),
            "mode": "xgb_anomalag_recency_recursive",
            "fit_start": str(FIT_START.date()),
            "n_recency": int(recency.sum()),
            "inner_iters": INNER_ITERS,
        }
    )
    prepared = {}
    pattern_prepared = {}
    for seed in runner.SEEDS:
        folds = []
        for fold in (0, 1):
            if fold not in prepared:
                prepared[fold] = runner.fold_data(train, states, fold)
            x_fit, y_fit, x_valid, y_valid, bfit, bvalid, cutoff, max_date = prepared[fold]
            pred = np.zeros(len(x_valid), dtype=np.float32)
            head_iters = {}
            for name in runner.BUCKETS:
                fit_idx, valid_idx = bfit == name, bvalid == name
                if fit_idx.sum() < 1000 or valid_idx.sum() < 1000:
                    runner.log({"stage": "skip_head", "seed": seed, "fold": fold, "bucket": name, "n_fit": int(fit_idx.sum()), "n_valid": int(valid_idx.sum())})
                    continue
                head = runner.make_model(seed, 700, 80)
                fit_target = runner.transform_targets(y_fit[fit_idx], x_fit.loc[fit_idx])
                valid_target = runner.transform_targets(y_valid[valid_idx], x_valid.loc[valid_idx])
                fit_head(head, x_fit.loc[fit_idx], fit_target, x_valid.loc[valid_idx], valid_target)
                transformed = head.predict(x_valid.loc[valid_idx]).astype(np.float32)
                pred[valid_idx] = runner.inverse_predictions(transformed, x_valid.loc[valid_idx])
                head_iters[name] = best_iters(head)
                del head
                gc.collect()
            valid_mask = np.asarray([not str(group).startswith("observed") for group in bvalid])
            row = {
                "stage": "fold",
                "seed": seed,
                "fold": fold,
                "head_iters": head_iters,
                "raw": runner.score(y_valid, pred, valid_mask),
                "period": [str(cutoff.date()), str(max_date.date())],
            }
            runner.log(row)
            folds.append(row)
        ordinary_mean = float(np.mean([r["raw"]["all"] for r in folds]))
        runner.log({"stage": "pair", "seed": seed, "mean": ordinary_mean})

        pattern_scores = []
        for anchor in ANCHORS:
            key = str(anchor)
            if key not in pattern_prepared:
                pattern_prepared[key] = pattern_fold_data(train, states, anchor)
            x_fit, y_fit, x_valid, y_valid, bfit, bvalid, exact, start, end = pattern_prepared[key]
            pred = np.zeros(len(x_valid), dtype=np.float32)
            for name in runner.BUCKETS:
                fit_idx, valid_idx = bfit == name, bvalid == name
                if fit_idx.sum() < 1000 or valid_idx.sum() < 200:
                    continue
                head = runner.make_model(seed, 700, 80)
                fit_head(
                    head,
                    x_fit.loc[fit_idx],
                    runner.transform_targets(y_fit[fit_idx], x_fit.loc[fit_idx]),
                    x_valid.loc[valid_idx],
                    runner.transform_targets(y_valid[valid_idx], x_valid.loc[valid_idx]),
                )
                pred[valid_idx] = runner.inverse_predictions(head.predict(x_valid.loc[valid_idx]).astype(np.float32), x_valid.loc[valid_idx])
                del head
                gc.collect()
            scores = runner.score(y_valid, pred, exact)
            runner.log({"stage": "pattern", "seed": seed, "anchor": key, "period": [start, end], "raw": scores})
            pattern_scores.append(scores["all"])
        if pattern_scores:
            runner.log({"stage": "pattern_mean", "seed": seed, "mean": float(np.mean(pattern_scores))})

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
            head = runner.make_model(seed, 500)
            fit_head(head, x_train.loc[fit_idx], runner.transform_targets(y_train[fit_idx], x_train.loc[fit_idx]))
            transformed = head.predict(x_test.loc[test_idx]).astype(np.float32)
            static_pred[test_idx] = runner.inverse_predictions(transformed, x_test.loc[test_idx])
            models[name] = head
        rec_pred = sequential_test_predict(models, train, train_input, test, y_train)
        for label, pred in (("static", static_pred), ("xgbanomalag", rec_pred)):
            runner.core.make_submission_frame(test, pred, sample).to_csv(
                runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_{label}_raw.csv",
                index=False,
            )
        for name, head in list(models.items()):
            del head
        gc.collect()
        runner.log({"stage": "full", "seed": seed, "ordinary_mean": ordinary_mean})
    runner.log({"stage": "complete"})
    while True:
        import time as time_mod

        time_mod.sleep(60)
    return 0


runner.core.build_features = build_features
runner.augment_supervised_features = augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.PREFIX = "xgb_horizon_idw_era5_anomalag"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (2027, 77)
runner.bucket = bucket
runner.make_model = model
raise SystemExit(custom_main())

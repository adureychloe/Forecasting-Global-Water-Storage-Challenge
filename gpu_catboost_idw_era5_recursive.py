"""IDW + ERA5 Ordered heads with recursive neighbourhood TWS at test time.

Training matches #226. Test walks months in order and writes own predictions
into later TWS_t so neighbourhood / lag features can use them. Fit-only ERA5
calendar-month anomalies are added in the augment hook.
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


def build_features(frame):
    features = enhanced_features.build_features(frame)
    features = neighborhood_tws_features.add_rich_neighborhood_features(frame, features)
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


def model(seed, iterations=500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=5,
        random_strength=1,
        grow_policy="SymmetricTree",
        boosting_type="Ordered",
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )


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
        pieces = [train_input, *history_test, month_raw]
        combined = pd.concat(pieces, ignore_index=True, sort=False)
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
        pred_all[pos] = pred_m
        filled = month_raw.copy()
        masked = filled["TWS_t_masked"].to_numpy(dtype=bool)
        tws = pd.to_numeric(filled["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
        tws = np.where(masked, pred_m, tws)
        filled["TWS_t"] = tws
        history_test.append(filled)
        runner.log({"stage": "recursive_month", "month": str(pd.Timestamp(month).date()), "n": int(len(month_raw))})
        gc.collect()
    return pred_all


def custom_main():
    train = pd.read_csv(runner.core.DATA / "Train.csv")
    test = pd.read_csv(runner.core.DATA / "Test.csv")
    sample = pd.read_csv(runner.core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {}
    runner.log({"stage": "start", "states": states.astype(int).tolist(), "buckets": list(runner.BUCKETS), "mode": "recursive_test"})
    y_train = train.target.to_numpy(np.float32)
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
                head = runner.make_model(seed)
                fit_target = runner.transform_targets(y_fit[fit_idx], x_fit.loc[fit_idx])
                valid_target = runner.transform_targets(y_valid[valid_idx], x_valid.loc[valid_idx])
                head.fit(x_fit.loc[fit_idx], fit_target, eval_set=(x_valid.loc[valid_idx], valid_target), early_stopping_rounds=80, verbose=False)
                transformed = head.predict(x_valid.loc[valid_idx]).astype(np.float32)
                pred[valid_idx] = runner.inverse_predictions(transformed, x_valid.loc[valid_idx])
                head_iters[name] = int(head.get_best_iteration() + 1)
                del head
                gc.collect()
            valid_mask = np.asarray([not str(group).startswith("observed") for group in bvalid])
            base_path = runner.RESULTS / f"fold{fold}" / "direct_base.npy"
            if base_path.exists():
                baseline = np.load(base_path)
                blends = {str(w): runner.score(y_valid, w * pred + (1 - w) * baseline, valid_mask) for w in np.arange(0.1, 1.01, 0.1)}
            else:
                blends = {"1.0": runner.score(y_valid, pred, valid_mask)}
            best = min(blends, key=lambda w: blends[w]["all"])
            row = {"stage": "fold", "seed": seed, "fold": fold, "head_iters": head_iters, "raw": runner.score(y_valid, pred, valid_mask), "best_weight": float(best), "best_blend": blends[best], "period": [str(cutoff.date()), str(max_date.date())]}
            runner.log(row)
            folds.append(row)
        mean = float(np.mean([r["best_blend"]["all"] for r in folds]))
        runner.log({"stage": "pair", "seed": seed, "mean": mean})

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
        models = {}
        static_pred = np.zeros(len(test), dtype=np.float32)
        for name in runner.BUCKETS:
            fit_idx, test_idx = bfit == name, btest == name
            if fit_idx.sum() < 1000 or test_idx.sum() == 0:
                models[name] = None
                continue
            head = runner.make_model(seed, 500)
            fit_target = runner.transform_targets(y_train[fit_idx], x_train.loc[fit_idx])
            head.fit(x_train.loc[fit_idx], fit_target, verbose=False)
            transformed = head.predict(x_test.loc[test_idx]).astype(np.float32)
            static_pred[test_idx] = runner.inverse_predictions(transformed, x_test.loc[test_idx])
            models[name] = head
        rec_pred = sequential_test_predict(models, train, train_input, test, y_train)
        for label, pred in (("static", static_pred), ("recursive", rec_pred)):
            runner.core.make_submission_frame(test, pred, sample).to_csv(
                runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_{label}_raw.csv",
                index=False,
            )
        for weight in (0.3, 0.4, 0.6):
            values = weight * rec_pred + (1 - weight) * static_pred
            runner.core.make_submission_frame(test, values, sample).to_csv(
                runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_recstat{weight:.1f}.csv",
                index=False,
            )
        for name, head in models.items():
            del head
        gc.collect()
        runner.log({"stage": "full", "seed": seed, "mean": mean})
    runner.log({"stage": "complete"})
    while True:
        import time as time_mod

        time_mod.sleep(60)
    return 0


runner.core.build_features = build_features
runner.augment_supervised_features = augment_features
runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.PREFIX = "catboost_horizon_idw_era5_recursive"
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"
runner.SEEDS = (2027, 77)
runner.bucket = bucket
runner.make_model = model
raise SystemExit(custom_main())

"""XGBoost GPU five-head horizon model under the official TWS mask schedule."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
PREFIX = "xgb_horizon5"
LOG = RESULTS / f"{PREFIX}.jsonl"
SEEDS = (77, 123)
BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")


def log(row):
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], BUCKETS[:4], default=BUCKETS[4])


def augment_supervised_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    """Optional fit-only feature hook; model wrappers may replace this function."""
    return x_fit, x_query


def score(y, p, masked):
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return {"all": rmse(y, p), "masked": rmse(y[masked], p[masked]), "observed": rmse(y[~masked], p[~masked])}


def make_model(seed, n_estimators=700):
    return XGBRegressor(
        n_estimators=n_estimators, max_depth=8, learning_rate=0.035,
        min_child_weight=12.0, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=20.0, reg_alpha=0.05, objective="reg:squarederror",
        eval_metric="rmse", tree_method="hist", device="cuda", max_bin=256,
        random_state=seed, n_jobs=16, early_stopping_rounds=80,
    )


def fold_data(train, states, fold):
    max_date = train.time.max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = train.loc[(train.time >= cutoff) & (train.time <= max_date)].copy()
    fit = train.loc[train.time < cutoff].copy()
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    combined = mask_by_template(combined, states, align_validation=True)
    features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    n = len(fit)
    y_fit = fit.target.to_numpy(np.float32)
    x_fit, x_valid = augment_supervised_features(
        features.iloc[:n].copy(), y_fit, features.iloc[n:].copy(), fit, valid
    )
    return x_fit, y_fit, x_valid, valid.target.to_numpy(np.float32), bucket(x_fit["TWS_t_age"].to_numpy(np.float32)), bucket(x_valid["TWS_t_age"].to_numpy(np.float32)), cutoff, max_date


def main():
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {}
    log({"stage": "start", "states": states.astype(int).tolist(), "buckets": BUCKETS})
    for seed in SEEDS:
        folds = []
        for fold in (0, 1):
            if fold not in prepared:
                prepared[fold] = fold_data(train, states, fold)
            xfit, yfit, xval, yval, bfit, bval, cutoff, max_date = prepared[fold]
            pred = np.zeros(len(xval), np.float32)
            iters = {}
            for name in BUCKETS:
                fi, vi = bfit == name, bval == name
                if fi.sum() < 1000 or vi.sum() < 1000:
                    continue
                model = make_model(seed)
                model.fit(xfit.loc[fi], yfit[fi], eval_set=[(xval.loc[vi], yval[vi])], verbose=False)
                pred[vi] = model.predict(xval.loc[vi]).astype(np.float32)
                iters[name] = int(model.best_iteration + 1)
                del model
                gc.collect()
            masked = bval != "observed"
            base = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
            blends = {str(w): score(yval, w * pred + (1 - w) * base, masked) for w in np.arange(0.1, 1.01, 0.1)}
            best = min(blends, key=lambda w: blends[w]["all"])
            row = {"stage": "fold", "seed": seed, "fold": fold, "head_iters": iters, "raw": score(yval, pred, masked), "best_weight": float(best), "best_blend": blends[best], "period": [str(cutoff.date()), str(max_date.date())]}
            log(row)
            folds.append(row)
        mean = float(np.mean([row["best_blend"]["all"] for row in folds]))
        log({"stage": "pair", "seed": seed, "mean": mean})
        train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
        test_input = test.copy()
        test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
        combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
        features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        xtrain, xtest = augment_supervised_features(
            features.iloc[:len(train)].copy(),
            train.target.to_numpy(np.float32),
            features.iloc[len(train):].copy(),
            train,
            test,
        )
        bfit = bucket(xtrain["TWS_t_age"].to_numpy(np.float32))
        btest = bucket(xtest["TWS_t_age"].to_numpy(np.float32))
        pred = np.zeros(len(test), np.float32)
        for name in BUCKETS:
            fi, ti = bfit == name, btest == name
            if fi.sum() < 1000 or ti.sum() == 0:
                continue
            model = make_model(seed, 500)
            model.set_params(early_stopping_rounds=None)
            model.fit(xtrain.loc[fi], train.target.to_numpy(np.float32)[fi], verbose=False)
            pred[ti] = model.predict(xtest.loc[ti]).astype(np.float32)
            del model
            gc.collect()
        base = pd.read_csv(SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(float)
        for weight in (0.3, 0.4, 0.6, 1.0):
            values = pred if weight == 1 else weight * pred + (1 - weight) * base
            suffix = "raw" if weight == 1 else f"blend{weight:.1f}"
            core.make_submission_frame(test, values, sample).to_csv(SUBMISSIONS / f"submission_{PREFIX}_seed{seed}_{suffix}.csv", index=False)
        log({"stage": "full", "seed": seed, "mean": mean})
    log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

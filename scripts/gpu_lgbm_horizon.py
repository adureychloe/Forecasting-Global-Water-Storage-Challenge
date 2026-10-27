"""LightGBM GPU age-routed horizon heads for a cross-family probe."""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

import gpu_horizon_runner as runner
import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (2027, 77)
runner.PREFIX = os.environ.get("LGBM_HORIZON_PREFIX", "lgbm_horizon")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], runner.BUCKETS[:4], default=runner.BUCKETS[4])


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y, p, mask): return {"all": rmse(y, p), "masked": rmse(y[mask], p[mask]), "observed": rmse(y[~mask], p[~mask])}


def augment_full_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    return runner.augment_supervised_features(
        x_fit, y_fit, x_query, fit_raw, query_raw
    )


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with runner.LOG.open("a", encoding="utf-8") as f: f.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def make_model(seed, n_estimators=700):
    return lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.03, num_leaves=63,
        max_depth=-1, min_child_samples=100, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=20.0, reg_alpha=0.1,
        objective="regression", metric="rmse", device_type="gpu",
        gpu_platform_id=0, gpu_device_id=0, max_bin=63,
        random_state=seed, n_jobs=16, verbosity=-1,
    )


runner.bucket = bucket


def main():
    train = pd.read_csv(core.DATA / "Train.csv"); test = pd.read_csv(core.DATA / "Test.csv"); sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time); test.time = pd.to_datetime(test.time); states = public_states(test)
    prepared = {}
    log({"stage": "start", "states": states.astype(int).tolist(), "buckets": list(runner.BUCKETS)})
    for seed in runner.SEEDS:
        folds = []
        for fold in (0, 1):
            if fold not in prepared: prepared[fold] = runner.fold_data(train, states, fold)
            xfit, yfit, xval, yval, bfit, bval, cutoff, max_date = prepared[fold]
            pred = np.zeros(len(xval), np.float32); iters = {}
            for name in runner.BUCKETS:
                fi, vi = bfit == name, bval == name
                if fi.sum() < 1000 or vi.sum() < 100: continue
                model = make_model(seed)
                model.fit(xfit.loc[fi], yfit[fi], eval_set=[(xval.loc[vi], yval[vi])], callbacks=[lgb.early_stopping(80, verbose=False)])
                pred[vi] = model.predict(xval.loc[vi], num_iteration=model.best_iteration_).astype(np.float32)
                iters[name] = int(model.best_iteration_ or 700); del model; gc.collect()
            mask = bval != "observed"; base = np.load(runner.RESULTS / f"fold{fold}" / "direct_base.npy")
            blends = {str(w): score(yval, w * pred + (1 - w) * base, mask) for w in np.arange(0.1, 1.01, 0.1)}; best = min(blends, key=lambda w: blends[w]["all"])
            row = {"stage": "fold", "seed": seed, "fold": fold, "head_iters": iters, "raw": score(yval, pred, mask), "best_weight": float(best), "best_blend": blends[best], "period": [str(cutoff.date()), str(max_date.date())]}; log(row); folds.append(row)
        mean = float(np.mean([x["best_blend"]["all"] for x in folds])); log({"stage": "pair", "seed": seed, "mean": mean})
        train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False); test_input = test.copy(); test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
        combined = pd.concat([train_input, test_input], ignore_index=True, sort=False); features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        xtrain, xtest = augment_full_features(
            features.iloc[:len(train)].copy(),
            train.target.to_numpy(np.float32),
            features.iloc[len(train):].copy(),
            train,
            test,
        ); btrain, btest = bucket(xtrain["TWS_t_age"].to_numpy()), bucket(xtest["TWS_t_age"].to_numpy()); pred = np.zeros(len(test), np.float32)
        for name in runner.BUCKETS:
            fi, ti = btrain == name, btest == name
            if fi.sum() < 1000 or ti.sum() == 0: continue
            model = make_model(seed, 500); model.fit(xtrain.loc[fi], train.target.to_numpy(np.float32)[fi]); pred[ti] = model.predict(xtest.loc[ti]).astype(np.float32); del model; gc.collect()
        base = pd.read_csv(runner.SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(float)
        for w in (0.3, 0.4, 0.6, 1.0):
            vals = pred if w == 1 else w * pred + (1 - w) * base; suffix = "raw" if w == 1 else f"blend{w:.1f}"
            core.make_submission_frame(test, vals, sample).to_csv(runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_{suffix}.csv", index=False)
        log({"stage": "full", "seed": seed, "mean": mean})
    log({"stage": "complete"}); return 0


if __name__ == "__main__": raise SystemExit(main())

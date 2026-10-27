"""Leakage-safe per-age blending calibration for horizon-head submissions."""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import pandas as pd

import gpu_horizon_runner as runner
import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

runner.BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
runner.SEEDS = (2027, 77)
runner.PREFIX = os.environ.get("HORIZON_CALIB_PREFIX", "horizon_age_calibrated")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def age_bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        runner.BUCKETS[:4],
        default=runner.BUCKETS[4],
    )


runner.bucket = age_bucket


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y, p, mask):
    return {"all": rmse(y, p), "masked": rmse(y[mask], p[mask]), "observed": rmse(y[~mask], p[~mask])}


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with runner.LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def make_model(seed, iterations=500):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=iterations, depth=8, learning_rate=0.04, l2_leaf_reg=10,
        random_strength=1, grow_policy="Lossguide", max_leaves=63,
        loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
        random_seed=seed, border_count=128, verbose=False, allow_writing_files=False,
    )


def main():
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {}
    weights_grid = np.arange(0.0, 1.01, 0.1)
    log({"stage": "start", "states": states.astype(int).tolist(), "buckets": list(runner.BUCKETS)})

    for seed in runner.SEEDS:
        fold_rows = []
        fold_predictions = []
        for fold in (0, 1):
            if fold not in prepared:
                prepared[fold] = runner.fold_data(train, states, fold)
            x_fit, y_fit, x_valid, y_valid, bfit, bvalid, cutoff, max_date = prepared[fold]
            pred = np.zeros(len(x_valid), dtype=np.float32)
            for name in runner.BUCKETS:
                fi, vi = bfit == name, bvalid == name
                if fi.sum() < 1000 or vi.sum() < 100:
                    continue
                model = make_model(seed)
                model.fit(x_fit.loc[fi], y_fit[fi], eval_set=(x_valid.loc[vi], y_valid[vi]), early_stopping_rounds=80, verbose=False)
                pred[vi] = model.predict(x_valid.loc[vi]).astype(np.float32)
                del model
                gc.collect()
            base = np.load(runner.RESULTS / f"fold{fold}" / "direct_base.npy")
            fold_predictions.append((y_valid, pred, base, bvalid))
            mask = bvalid != "observed"
            global_blends = {str(w): score(y_valid, w * pred + (1 - w) * base, mask) for w in weights_grid}
            global_best = min(global_blends, key=lambda w: global_blends[w]["all"])
            fold_rows.append({"fold": fold, "global_weight": float(global_best), "global": global_blends[global_best], "period": [str(cutoff.date()), str(max_date.date())]})

        weights = {}
        for name in runner.BUCKETS:
            ys, ps, bs = [], [], []
            for y, p, b, bucket in fold_predictions:
                keep = bucket == name
                if keep.any():
                    ys.append(y[keep]); ps.append(p[keep]); bs.append(b[keep])
            if not ys:
                weights[name] = 0.3
                continue
            y_all, p_all, b_all = np.concatenate(ys), np.concatenate(ps), np.concatenate(bs)
            candidate = {float(w): rmse(y_all, w * p_all + (1 - w) * b_all) for w in weights_grid}
            weights[name] = float(min(candidate, key=candidate.get))

        blended_all = []
        for y, p, b, bucket in fold_predictions:
            w = np.array([weights[x] for x in bucket], dtype=np.float32)
            blended_all.append((y, w * p + (1 - w) * b, bucket != "observed"))
        ycat = np.concatenate([x[0] for x in blended_all])
        pcat = np.concatenate([x[1] for x in blended_all])
        mcat = np.concatenate([x[2] for x in blended_all])
        log({"stage": "pair", "seed": seed, "weights": weights, "calibrated": score(ycat, pcat, mcat), "folds": fold_rows})

        train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
        test_input = test.copy()
        test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
        combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
        features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        x_train, x_test = features.iloc[:len(train)], features.iloc[len(train):]
        bfit = age_bucket(x_train["TWS_t_age"].to_numpy(np.float32))
        btest = age_bucket(x_test["TWS_t_age"].to_numpy(np.float32))
        pred = np.zeros(len(test), dtype=np.float32)
        for name in runner.BUCKETS:
            fi, ti = bfit == name, btest == name
            if fi.sum() < 1000 or ti.sum() == 0:
                continue
            model = make_model(seed)
            model.fit(x_train.loc[fi], train.target.to_numpy(np.float32)[fi], verbose=False)
            pred[ti] = model.predict(x_test.loc[ti]).astype(np.float32)
            del model
            gc.collect()
        base = pd.read_csv(runner.SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(float)
        wtest = np.array([weights[x] for x in btest], dtype=np.float32)
        values = wtest * pred + (1 - wtest) * base
        runner.core.make_submission_frame(test, values, sample).to_csv(runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_calibrated.csv", index=False)
        for w in (0.3, 0.4, 0.6):
            fixed = w * pred + (1 - w) * base
            runner.core.make_submission_frame(test, fixed, sample).to_csv(runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_fixed{w:.1f}.csv", index=False)
        log({"stage": "full", "seed": seed, "weights": weights})
    log({"stage": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

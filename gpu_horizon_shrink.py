"""Age-conditioned mean-centering/shrinkage post-process for horizon heads."""

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
runner.PREFIX = os.environ.get("HORIZON_SHRINK_PREFIX", "horizon_shrink")
runner.LOG = runner.RESULTS / f"{runner.PREFIX}.jsonl"


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], runner.BUCKETS[:4], default=runner.BUCKETS[4])


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y, p, mask):
    return {"all": rmse(y, p), "masked": rmse(y[mask], p[mask]), "observed": rmse(y[~mask], p[~mask])}


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with runner.LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def model_for(seed, iterations=500):
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
    alphas = np.arange(0.25, 1.51, 0.25)
    blend_weights = np.arange(0.0, 1.01, 0.1)
    log({"stage": "start", "states": states.astype(int).tolist(), "buckets": list(runner.BUCKETS)})

    for seed in runner.SEEDS:
        records = []
        for fold in (0, 1):
            if fold not in prepared:
                prepared[fold] = runner.fold_data(train, states, fold)
            xfit, yfit, xval, yval, bfit, bval, cutoff, max_date = prepared[fold]
            pred = np.zeros(len(xval), dtype=np.float32)
            mus = {name: float(np.mean(yfit[bfit == name])) for name in runner.BUCKETS if np.any(bfit == name)}
            for name in runner.BUCKETS:
                fi, vi = bfit == name, bval == name
                if fi.sum() < 1000 or vi.sum() < 100:
                    continue
                m = model_for(seed)
                m.fit(xfit.loc[fi], yfit[fi], eval_set=(xval.loc[vi], yval[vi]), early_stopping_rounds=80, verbose=False)
                pred[vi] = m.predict(xval.loc[vi]).astype(np.float32)
                del m
                gc.collect()
            base = np.load(runner.RESULTS / f"fold{fold}" / "direct_base.npy")
            records.append((yval, pred, base, bval, mus))
            log({"stage": "fold", "seed": seed, "fold": fold, "raw": score(yval, pred, bval != "observed"), "period": [str(cutoff.date()), str(max_date.date())]})

        params = {}
        for name in runner.BUCKETS:
            best = (float("inf"), 1.0, 0.5)
            for alpha in alphas:
                for blend in blend_weights:
                    ys, ps = [], []
                    for y, p, base, bval, mus in records:
                        keep = bval == name
                        if keep.any():
                            shrunk = mus.get(name, float(np.mean(y))) + alpha * (p[keep] - mus.get(name, float(np.mean(y))))
                            ys.append(y[keep]); ps.append(blend * shrunk + (1 - blend) * base[keep])
                    if ys:
                        value = rmse(np.concatenate(ys), np.concatenate(ps))
                        if value < best[0]:
                            best = (value, float(alpha), float(blend))
            params[name] = {"alpha": best[1], "blend": best[2], "rmse": best[0]}
        outs = []
        for y, p, base, bval, mus in records:
            out = np.zeros_like(p)
            for name in runner.BUCKETS:
                keep = bval == name
                if keep.any():
                    par = params[name]
                    shrunk = mus.get(name, float(np.mean(y))) + par["alpha"] * (p[keep] - mus.get(name, float(np.mean(y))))
                    out[keep] = par["blend"] * shrunk + (1 - par["blend"]) * base[keep]
            outs.append((y, out, bval != "observed"))
        yall = np.concatenate([x[0] for x in outs]); pall = np.concatenate([x[1] for x in outs]); mall = np.concatenate([x[2] for x in outs])
        fold_mean = float(np.mean([score(y, p, m)["all"] for y, p, m in outs]))
        log({"stage": "pair", "seed": seed, "params": params, "fold_mean": fold_mean, "pooled": score(yall, pall, mall)})

        train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
        test_input = test.copy(); test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
        combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
        features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        xtrain, xtest = features.iloc[:len(train)], features.iloc[len(train):]
        btrain, btest = bucket(xtrain["TWS_t_age"].to_numpy(np.float32)), bucket(xtest["TWS_t_age"].to_numpy(np.float32))
        pred = np.zeros(len(test), dtype=np.float32)
        mus_full = {name: float(np.mean(train.target.to_numpy(np.float32)[btrain == name])) for name in runner.BUCKETS if np.any(btrain == name)}
        for name in runner.BUCKETS:
            fi, ti = btrain == name, btest == name
            if fi.sum() < 1000 or ti.sum() == 0: continue
            m = model_for(seed); m.fit(xtrain.loc[fi], train.target.to_numpy(np.float32)[fi], verbose=False)
            pred[ti] = m.predict(xtest.loc[ti]).astype(np.float32); del m; gc.collect()
        base = pd.read_csv(runner.SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(float)
        values = np.zeros(len(test), dtype=float)
        for name in runner.BUCKETS:
            keep = btest == name; par = params[name]; center = mus_full.get(name, float(np.mean(train.target)))
            values[keep] = par["blend"] * (center + par["alpha"] * (pred[keep] - center)) + (1 - par["blend"]) * base[keep]
        core.make_submission_frame(test, values, sample).to_csv(runner.SUBMISSIONS / f"submission_{runner.PREFIX}_seed{seed}_calibrated.csv", index=False)
        log({"stage": "full", "seed": seed, "params": params})
    log({"stage": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

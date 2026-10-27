"""Public-pattern validation of age-bucket mean shrinkage."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOG = RESULTS / "pattern_shrink_alpha.jsonl"
BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
ANCHORS = (pd.Period("2007-09", freq="M"), pd.Period("2009-01", freq="M"))
SEEDS = (2027, 77, 123)
ALPHAS = np.arange(0.5, 1.51, 0.25)


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], BUCKETS[:4], default=BUCKETS[4])


def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as f: f.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def main():
    train = pd.read_csv(DATA / "Train.csv"); train.time = pd.to_datetime(train.time)
    test = pd.read_csv(DATA / "Test.csv", usecols=["time", "TWS_t_masked"]); test.time = pd.to_datetime(test.time)
    states = public_states(test); months_all = train.time.dt.to_period("M")
    log({"stage": "start", "anchors": [str(a) for a in ANCHORS], "seeds": SEEDS})
    for seed in SEEDS:
        for anchor in ANCHORS:
            months = pd.PeriodIndex([anchor + i for i in range(len(states))], freq="M")
            valid = train.loc[months_all.isin(months)].copy(); fit = train.loc[months_all < anchor].tail(500_000).copy(); smap = dict(zip(months, states))
            vm = valid.time.dt.to_period("M").map(smap).to_numpy(bool); combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
            combined = mask_by_template(combined, states, align_validation=False); n = len(fit); combined.loc[n:, "TWS_t_masked"] = vm; combined.loc[n + np.flatnonzero(vm), "TWS_t"] = np.nan
            features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); xfit, xval = features.iloc[:n], features.iloc[n:]; yfit, yval = fit.target.to_numpy(np.float32), valid.target.to_numpy(np.float32); bfit, bval = bucket(xfit["TWS_t_age"].to_numpy()), bucket(xval["TWS_t_age"].to_numpy()); pred = np.zeros(len(valid), np.float32); params = {}
            for name in BUCKETS:
                fi, vi = bfit == name, bval == name
                if fi.sum() < 1000 or vi.sum() < 100: continue
                m = CatBoostRegressor(iterations=500, depth=8, learning_rate=0.04, l2_leaf_reg=10, random_strength=1, grow_policy="Lossguide", max_leaves=63, loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0", random_seed=seed, border_count=128, verbose=False, allow_writing_files=False)
                m.fit(xfit.loc[fi], yfit[fi], eval_set=(xval.loc[vi], yval[vi]), early_stopping_rounds=80, verbose=False); pred[vi] = m.predict(xval.loc[vi]).astype(np.float32); del m; gc.collect()
                center = float(np.mean(yfit[fi])); best = (float("inf"), 1.0)
                for alpha in ALPHAS:
                    value = rmse(yval[vi], center + alpha * (pred[vi] - center))
                    if value < best[0]: best = (value, float(alpha))
                params[name] = {"alpha": best[1], "raw_rmse": rmse(yval[vi], pred[vi]), "shrink_rmse": best[0]}
            raw = rmse(yval, pred); out = pred.copy()
            for name, par in params.items():
                keep = bval == name; center = float(np.mean(yfit[bfit == name])); out[keep] = center + par["alpha"] * (pred[keep] - center)
            log({"stage": "fold", "seed": seed, "anchor": str(anchor), "raw": raw, "shrink": rmse(yval, out), "masked_raw": rmse(yval[vm], pred[vm]), "masked_shrink": rmse(yval[vm], out[vm]), "params": params, "rows": len(valid)})
    log({"stage": "complete"}); return 0


if __name__ == "__main__": raise SystemExit(main())

"""Pattern-matched CV: replay the 18 public test month states on historical blocks."""

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
RESULTS = ROOT / "results"
LOG = RESULTS / "pattern_cv.jsonl"
BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
SEEDS = (77, 123, 2027)
ANCHORS = (pd.Period("2007-09", freq="M"), pd.Period("2009-01", freq="M"))


def variant_bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], BUCKETS[:4], default=BUCKETS[4])


def log(row):
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def model(seed):
    return CatBoostRegressor(
        iterations=500, depth=8, learning_rate=0.04, l2_leaf_reg=10,
        random_strength=1, grow_policy="Lossguide", max_leaves=63,
        loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
        random_seed=seed, border_count=128, verbose=False, allow_writing_files=False,
    )


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fold_data(train, states, anchor):
    test_months = pd.PeriodIndex([anchor + i for i in range(len(states))], freq="M")
    all_months = train.time.dt.to_period("M")
    valid_mask = all_months.isin(test_months)
    valid = train.loc[valid_mask].copy()
    fit = train.loc[(all_months < anchor)].tail(500_000).copy()
    valid_month = valid.time.dt.to_period("M")
    state_map = dict(zip(test_months, states))
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    combined = mask_by_template(combined, states, align_validation=False)
    n = len(fit)
    exact = valid_month.map(state_map).to_numpy(dtype=bool)
    combined.loc[n:, "TWS_t_masked"] = exact
    combined.loc[n + np.flatnonzero(exact), "TWS_t"] = np.nan
    features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    ages = features["TWS_t_age"].to_numpy(dtype=np.float32)
    return features.iloc[:n], fit.target.to_numpy(np.float32), features.iloc[n:], valid.target.to_numpy(np.float32), variant_bucket(ages[:n]), variant_bucket(ages[n:]), exact, str(test_months[0]), str(test_months[-1])


def main():
    train = pd.read_csv(core.DATA / "Train.csv")
    train.time = pd.to_datetime(train.time)
    test = pd.read_csv(core.DATA / "Test.csv", usecols=["time", "TWS_t_masked"])
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {str(a): fold_data(train, states, a) for a in ANCHORS}
    log({"stage": "start", "anchors": [str(a) for a in ANCHORS], "states": states.astype(int).tolist()})
    for seed in SEEDS:
        for anchor in ANCHORS:
            xfit, yfit, xval, yval, bfit, bval, valid_mask, start, end = prepared[str(anchor)]
            pred = np.zeros(len(xval), dtype=np.float32)
            head_iters = {}
            for name in BUCKETS:
                fi, vi = bfit == name, bval == name
                if fi.sum() < 1000 or vi.sum() < 1000:
                    continue
                m = model(seed)
                m.fit(xfit.loc[fi], yfit[fi], eval_set=(xval.loc[vi], yval[vi]), early_stopping_rounds=80, verbose=False)
                pred[vi] = m.predict(xval.loc[vi]).astype(np.float32)
                head_iters[name] = int(m.get_best_iteration() + 1)
                del m
                gc.collect()
            raw = rmse(yval, pred)
            log({"stage": "fold", "seed": seed, "anchor": str(anchor), "period": [start, end], "head_iters": head_iters, "raw": raw, "masked": rmse(yval[valid_mask], pred[valid_mask]), "observed": rmse(yval[~valid_mask], pred[~valid_mask]), "rows": len(yval)})
    log({"stage": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

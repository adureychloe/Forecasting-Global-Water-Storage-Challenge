"""Exact sparse-calendar replay comparing 12- and 40-month frozen blocks."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import enhanced_features
import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states
import target_stats_blockfrozen_features
import target_stats_features


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOG = RESULTS / "pattern_blockspan_cv.jsonl"
BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")
ANCHORS = (pd.Period("2007-09", freq="M"), pd.Period("2009-01", freq="M"))
SEED = 2027


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def bucket(age):
    return np.select(
        [age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5],
        BUCKETS[:4],
        default=BUCKETS[4],
    )


def rmse(target, prediction):
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def model():
    return CatBoostRegressor(
        iterations=500,
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
        random_seed=SEED,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )


def sparse_months(test, anchor):
    periods = pd.PeriodIndex(pd.to_datetime(test["time"]).sort_values().unique(), freq="M")
    offsets = periods.astype("int64") - int(periods.min().ordinal)
    return pd.PeriodIndex([anchor + int(offset) for offset in offsets], freq="M")


def prepared(train, test, states, anchor):
    months = sparse_months(test, anchor)
    all_months = train.time.dt.to_period("M")
    valid = train.loc[all_months.isin(months)].copy()
    fit = train.loc[all_months < anchor].copy()
    state_map = dict(zip(months, states))
    valid_mask = valid.time.dt.to_period("M").map(state_map).to_numpy(dtype=bool)
    combined = pd.concat(
        [fit.drop(columns=["target"]), valid.drop(columns=["target"])],
        ignore_index=True,
        sort=False,
    )
    combined = mask_by_template(combined, states, align_validation=False)
    n = len(fit)
    combined.loc[n:, "TWS_t_masked"] = valid_mask
    combined.loc[n + np.flatnonzero(valid_mask), "TWS_t"] = np.nan
    features = (
        enhanced_features.build_features(combined)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )
    return fit, valid, features.iloc[:n].copy(), features.iloc[n:].copy(), valid_mask, months


def add_block_features(x_fit, y_fit, x_valid, fit, valid, block_months):
    x_fit, x_valid = target_stats_features.augment_features(
        x_fit, y_fit, x_valid, fit, valid
    )
    fit_extra = target_stats_blockfrozen_features._training_block_features(
        fit, y_fit, valid, block_months=block_months
    )
    query_extra = target_stats_blockfrozen_features._query_block_features(
        fit, y_fit, valid
    )
    for column in target_stats_blockfrozen_features.FEATURES:
        x_fit[column] = fit_extra[column].fillna(0).to_numpy(dtype=np.float32)
        x_valid[column] = query_extra[column].fillna(0).to_numpy(dtype=np.float32)
    return x_fit, x_valid


def evaluate(x_fit, y_fit, x_valid, y_valid, valid_mask):
    fit_bucket = bucket(x_fit["TWS_t_age"].to_numpy(dtype=np.float32))
    valid_bucket = bucket(x_valid["TWS_t_age"].to_numpy(dtype=np.float32))
    prediction = np.zeros(len(x_valid), dtype=np.float32)
    iterations = {}
    for name in BUCKETS:
        fit_idx, valid_idx = fit_bucket == name, valid_bucket == name
        if fit_idx.sum() < 1000 or valid_idx.sum() < 1000:
            continue
        estimator = model()
        estimator.fit(
            x_fit.loc[fit_idx],
            y_fit[fit_idx],
            eval_set=(x_valid.loc[valid_idx], y_valid[valid_idx]),
            early_stopping_rounds=80,
            verbose=False,
        )
        prediction[valid_idx] = estimator.predict(x_valid.loc[valid_idx]).astype(np.float32)
        iterations[name] = int(estimator.get_best_iteration() + 1)
        del estimator
        gc.collect()
    return {
        "all": rmse(y_valid, prediction),
        "masked": rmse(y_valid[valid_mask], prediction[valid_mask]),
        "observed": rmse(y_valid[~valid_mask], prediction[~valid_mask]),
        "head_iters": iterations,
    }


def main():
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv", usecols=["time", "TWS_t_masked"])
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    log({"stage": "start", "anchors": [str(anchor) for anchor in ANCHORS], "states": states.astype(int).tolist()})
    for anchor in ANCHORS:
        fit, valid, base_fit, base_valid, valid_mask, months = prepared(
            train, test, states, anchor
        )
        y_fit = fit.target.to_numpy(dtype=np.float32)
        y_valid = valid.target.to_numpy(dtype=np.float32)
        for block_months in (12, 40):
            x_fit, x_valid = add_block_features(
                base_fit.copy(),
                y_fit,
                base_valid.copy(),
                fit,
                valid,
                block_months,
            )
            scores = evaluate(x_fit, y_fit, x_valid, y_valid, valid_mask)
            log(
                {
                    "stage": "fold",
                    "anchor": str(anchor),
                    "block_months": block_months,
                    "period": [str(months.min()), str(months.max())],
                    "rows": len(valid),
                    **scores,
                }
            )
            del x_fit, x_valid
            gc.collect()
        del fit, valid, base_fit, base_valid
        gc.collect()
    log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

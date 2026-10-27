"""Horizon/age-bucket direct heads layered on the official mask schedule."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
PREFIX = os.environ.get("HORIZON_PREFIX", "horizon_heads")
LOG = RESULTS / f"{PREFIX}.jsonl"
SEEDS = (2027, 77)
BUCKETS = ("observed", "short", "mid", "long")


def log(row: dict) -> None:
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def bucket(age: np.ndarray) -> np.ndarray:
    return np.select([age <= 0.5, age <= 2.5, age <= 6.5], BUCKETS[:3], default=BUCKETS[3])


def augment_supervised_features(x_fit, y_fit, x_query, fit_raw, query_raw):
    """Optional fit-only feature hook; model wrappers may replace this function."""
    return x_fit, x_query


def transform_targets(target, features):
    """Optional target-space hook; wrappers may replace this function."""
    return np.asarray(target, dtype=np.float32)


def inverse_predictions(prediction, features):
    """Invert a wrapper-specific target transform in prediction space."""
    return np.asarray(prediction, dtype=np.float32)


def route_groups(x_fit, x_query, fit_raw, query_raw):
    """Optional routing hook; defaults to the configured TWS-age buckets."""
    return (
        bucket(x_fit["TWS_t_age"].to_numpy(dtype=np.float32)),
        bucket(x_query["TWS_t_age"].to_numpy(dtype=np.float32)),
    )


def fold_data(train: pd.DataFrame, states: np.ndarray, fold: int):
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
    bfit, bvalid = route_groups(x_fit, x_valid, fit, valid)
    return x_fit, y_fit, x_valid, valid.target.to_numpy(np.float32), bfit, bvalid, cutoff, max_date


def make_model(seed: int, iterations: int = 500) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations, depth=8, learning_rate=0.04, l2_leaf_reg=10,
        random_strength=1, grow_policy="Lossguide", max_leaves=63,
        loss_function="RMSE", eval_metric="RMSE", task_type="GPU", devices="0",
        random_seed=seed, border_count=128, verbose=False, allow_writing_files=False,
    )


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y, p, mask):
    return {"all": rmse(y, p), "masked": rmse(y[mask], p[mask]), "observed": rmse(y[~mask], p[~mask])}


def main() -> int:
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {}
    full_prepared = None
    log({"stage": "start", "states": states.astype(int).tolist(), "buckets": list(BUCKETS)})
    for seed in SEEDS:
        folds = []
        for fold in (0, 1):
            if fold not in prepared:
                prepared[fold] = fold_data(train, states, fold)
            x_fit, y_fit, x_valid, y_valid, bfit, bvalid, cutoff, max_date = prepared[fold]
            pred = np.zeros(len(x_valid), dtype=np.float32)
            head_iters = {}
            for name in BUCKETS:
                fit_idx, valid_idx = bfit == name, bvalid == name
                if fit_idx.sum() < 1000 or valid_idx.sum() < 1000:
                    log({"stage": "skip_head", "seed": seed, "fold": fold, "bucket": name, "n_fit": int(fit_idx.sum()), "n_valid": int(valid_idx.sum())})
                    continue
                model = make_model(seed)
                fit_target = transform_targets(y_fit[fit_idx], x_fit.loc[fit_idx])
                valid_target = transform_targets(y_valid[valid_idx], x_valid.loc[valid_idx])
                model.fit(x_fit.loc[fit_idx], fit_target, eval_set=(x_valid.loc[valid_idx], valid_target), early_stopping_rounds=80, verbose=False)
                transformed = model.predict(x_valid.loc[valid_idx]).astype(np.float32)
                pred[valid_idx] = inverse_predictions(transformed, x_valid.loc[valid_idx])
                head_iters[name] = int(model.get_best_iteration() + 1)
                del model
                gc.collect()
            valid_mask = np.asarray([not str(group).startswith("observed") for group in bvalid])
            baseline = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
            blends = {str(w): score(y_valid, w * pred + (1 - w) * baseline, valid_mask) for w in np.arange(0.1, 1.01, 0.1)}
            best = min(blends, key=lambda w: blends[w]["all"])
            row = {"stage": "fold", "seed": seed, "fold": fold, "head_iters": head_iters, "raw": score(y_valid, pred, valid_mask), "best_weight": float(best), "best_blend": blends[best], "period": [str(cutoff.date()), str(max_date.date())]}
            log(row)
            folds.append(row)
        if len(folds) != 2:
            continue
        mean = float(np.mean([r["best_blend"]["all"] for r in folds]))
        log({"stage": "pair", "seed": seed, "mean": mean})
        # Full candidates deliberately use fixed, compact 500-iteration heads;
        # validation is used only to decide whether to retain the family.
        if full_prepared is None:
            train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
            test_input = test.copy()
            test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
            combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
            features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            x_train, x_test = augment_supervised_features(
                features.iloc[:len(train)].copy(),
                train.target.to_numpy(np.float32),
                features.iloc[len(train):].copy(),
                train,
                test,
            )
            bfit, btest = route_groups(x_train, x_test, train, test)
            fit_rows = int(os.environ.get("HORIZON_FIT_ROWS", "0"))
            fit_selector = np.ones(len(x_train), dtype=bool)
            if fit_rows > 0 and fit_rows < len(x_train):
                fit_selector[:-fit_rows] = False
            full_prepared = x_train, x_test, bfit, btest, fit_selector
        else:
            x_train, x_test, bfit, btest, fit_selector = full_prepared
        pred = np.zeros(len(test), dtype=np.float32)
        for name in BUCKETS:
            fit_idx, test_idx = (bfit == name) & fit_selector, btest == name
            if fit_idx.sum() < 1000 or test_idx.sum() == 0:
                continue
            model = make_model(seed, 500)
            fit_target = transform_targets(train.target.to_numpy(np.float32)[fit_idx], x_train.loc[fit_idx])
            model.fit(x_train.loc[fit_idx], fit_target, verbose=False)
            transformed = model.predict(x_test.loc[test_idx]).astype(np.float32)
            pred[test_idx] = inverse_predictions(transformed, x_test.loc[test_idx])
            del model
            gc.collect()
        base = pd.read_csv(SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(dtype=float)
        for weight in (0.3, 0.4, 0.6, 1.0):
            values = pred if weight == 1 else weight * pred + (1 - weight) * base
            suffix = "raw" if weight == 1 else f"blend{weight:.1f}"
            core.make_submission_frame(test, values, sample).to_csv(SUBMISSIONS / f"submission_{PREFIX}_seed{seed}_{suffix}.csv", index=False)
        log({"stage": "full", "seed": seed, "mean": mean})
    log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

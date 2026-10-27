"""Leakage-safe two-head direct target model for observed vs masked TWS rows."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import gpu_nightly_runner as core

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
LOG = RESULTS / "dualhead.jsonl"
SEEDS = (42, 77, 2027)


def log(row: dict) -> None:
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def public_states(test: pd.DataFrame) -> np.ndarray:
    rates = test.groupby("time", sort=True)["TWS_t_masked"].mean().to_numpy(dtype=float)
    return rates >= 0.5


def mask_by_template(frame: pd.DataFrame, states: np.ndarray, *, align_validation: bool) -> pd.DataFrame:
    result = frame.copy()
    dates = np.sort(result["time"].unique())
    n = len(dates)
    period = len(states)
    # Validation ends at the last template state; full train ends just before
    # the first public test state, so its latest synthetic state is state[-1].
    offset = n - period if align_validation else n + 1
    indexes = (np.arange(n) - offset) % period
    states_by_date = dict(zip(dates, states[indexes]))
    masked = result["time"].map(states_by_date).to_numpy(dtype=bool)
    result.loc[masked, "TWS_t"] = np.nan
    result["TWS_t_masked"] = masked
    return result


def make_fold(train: pd.DataFrame, states: np.ndarray, fold: int):
    max_date = train["time"].max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = train.loc[(train.time >= cutoff) & (train.time <= max_date)].copy()
    fit = train.loc[train.time < cutoff].copy()
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    combined = mask_by_template(combined, states, align_validation=True)
    features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    n_fit = len(fit)
    fit_masked = combined.TWS_t_masked.iloc[:n_fit].to_numpy(dtype=bool)
    valid_masked = combined.TWS_t_masked.iloc[n_fit:].to_numpy(dtype=bool)
    return features.iloc[:n_fit], fit.target.to_numpy(np.float32), features.iloc[n_fit:], valid.target.to_numpy(np.float32), fit_masked, valid_masked, cutoff, max_date


def make_model(seed: int, iterations: int = 1800) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=10,
        random_strength=1,
        grow_policy="Lossguide",
        max_leaves=63,
        loss_function="RMSE",
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=seed,
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "all": rmse(y, p),
        "masked": rmse(y[mask], p[mask]),
        "observed": rmse(y[~mask], p[~mask]),
    }


def fit_head(seed: int, x_train: pd.DataFrame, y_train: np.ndarray, x_valid: pd.DataFrame, y_valid: np.ndarray):
    model = make_model(seed)
    model.fit(x_train, y_train, eval_set=(x_valid, y_valid), early_stopping_rounds=150, verbose=False)
    return model, int(model.get_best_iteration() + 1)


def full_features(train: pd.DataFrame, test: pd.DataFrame, states: np.ndarray):
    train_inputs = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
    test_inputs = test.copy()
    test_inputs.loc[test_inputs.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
    combined = pd.concat([train_inputs, test_inputs], ignore_index=True, sort=False)
    features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    train_masked = train_inputs["TWS_t_masked"].to_numpy(dtype=bool)
    test_masked = test_inputs["TWS_t_masked"].to_numpy(dtype=bool)
    return features.iloc[:len(train)], train.target.to_numpy(np.float32), features.iloc[len(train):], train_masked, test_masked


def prior_fold_rows() -> dict[tuple[int, int], dict]:
    latest: dict[tuple[int, int], dict] = {}
    if not LOG.exists():
        return latest
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "fold":
                latest[(int(row["seed"]), int(row["fold"]))] = row
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return latest


def prior_full_seeds() -> set[int]:
    done: set[int] = set()
    if not LOG.exists():
        return done
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "full":
                done.add(int(row["seed"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return done


def main() -> int:
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {}
    previous = prior_fold_rows()
    full_done = prior_full_seeds()
    log({"stage": "start", "states": states.astype(int).tolist()})
    for seed in SEEDS:
        fold_rows = []
        for fold in (0, 1):
            if (seed, fold) in previous:
                fold_rows.append(previous[(seed, fold)])
                continue
            if fold not in prepared:
                prepared[fold] = make_fold(train, states, fold)
            x_fit, y_fit, x_valid, y_valid, fit_masked, valid_masked, cutoff, max_date = prepared[fold]
            predictions = np.zeros(len(x_valid), dtype=np.float32)
            iterations = []
            for name, mask_value in (("masked", True), ("observed", False)):
                train_idx = fit_masked == mask_value
                valid_idx = valid_masked == mask_value
                if train_idx.sum() < 1000 or valid_idx.sum() < 1000:
                    continue
                head, best_iteration = fit_head(seed, x_fit.loc[train_idx], y_fit[train_idx], x_valid.loc[valid_idx], y_valid[valid_idx])
                predictions[valid_idx] = head.predict(x_valid.loc[valid_idx]).astype(np.float32)
                iterations.append(best_iteration)
                del head
                gc.collect()
            baseline = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
            blends = {str(w): score(y_valid, w * predictions + (1 - w) * baseline, valid_masked) for w in np.arange(0.1, 1.01, 0.1)}
            best_weight = min(blends, key=lambda w: blends[w]["all"])
            row = {"stage": "fold", "seed": seed, "fold": fold, "iterations": int(round(np.mean(iterations))), "raw": score(y_valid, predictions, valid_masked), "best_weight": float(best_weight), "best_blend": blends[best_weight], "period": [str(cutoff.date()), str(max_date.date())]}
            log(row)
            fold_rows.append(row)
        if len(fold_rows) != 2:
            continue
        mean = float(np.mean([row["best_blend"]["all"] for row in fold_rows]))
        iterations = max(100, int(round(np.mean([row["iterations"] for row in fold_rows]))))
        log({"stage": "pair", "seed": seed, "mean": mean, "iterations": iterations})
        if seed in full_done:
            continue
        x_train, y_train, x_test, train_masked, test_masked = full_features(train, test, states)
        prediction = np.zeros(len(x_test), dtype=np.float32)
        for name, mask_value in (("masked", True), ("observed", False)):
            train_idx = train_masked == mask_value
            test_idx = test_masked == mask_value
            head = make_model(seed, iterations)
            head.fit(x_train.loc[train_idx], y_train[train_idx], verbose=False)
            prediction[test_idx] = head.predict(x_test.loc[test_idx]).astype(np.float32)
            del head
            gc.collect()
        base = pd.read_csv(SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(dtype=float)
        for weight in (0.2, 0.3, 0.4, 0.5, 0.6, 1.0):
            values = prediction if weight == 1 else weight * prediction + (1 - weight) * base
            suffix = "raw" if weight == 1 else f"blend{weight:.1f}"
            core.make_submission_frame(test, values, sample).to_csv(SUBMISSIONS / f"submission_dualhead_seed{seed}_{suffix}.csv", index=False)
        log({"stage": "full", "seed": seed, "mean": mean, "iterations": iterations})
        del x_train, y_train, x_test
        gc.collect()
    log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

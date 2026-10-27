"""GPU experiment: emulate the public test's consecutive TWS masking pattern.

The existing mask-90 route makes independent row-level masks.  The competition
test instead has public, almost all-or-nothing month states and therefore long
per-cell gaps.  This runner tiles that state sequence backward over historical
dates for training and tail validation.  It never uses a future TWS value.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import gpu_nightly_runner as core


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
LOG = RESULTS / "sequence_mask.jsonl"
SEEDS = (42, 77, 2027)
CONFIGS = (
    ("seq_lossguide63_direct", "direct"),
    ("seq_lossguide63_deltaffill", "delta_ffill"),
)


def write_log(row: dict) -> None:
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def template_states(test: pd.DataFrame) -> np.ndarray:
    grouped = test.groupby("time", sort=True)["TWS_t_masked"].mean()
    return (grouped.to_numpy(dtype=float) >= 0.5)


def apply_historical_template(frame: pd.DataFrame, states: np.ndarray, *, end_at_template_end: bool) -> pd.DataFrame:
    """Mask entire dates with a cyclic copy of public test states.

    Validation is anchored so its latest date has the latest test state.  For
    a final fit, training is anchored immediately before the public test starts.
    Both cases make forward-fill histories available only from earlier dates.
    """
    result = frame.copy()
    dates = np.sort(result["time"].unique())
    n_dates, period = len(dates), len(states)
    if end_at_template_end:
        indexes = (np.arange(n_dates) - (n_dates - period)) % period
    else:
        indexes = (np.arange(n_dates) - n_dates) % period
    by_date = dict(zip(dates, states[indexes]))
    masked = result["time"].map(by_date).to_numpy(dtype=bool)
    result.loc[masked, "TWS_t"] = np.nan
    result["TWS_t_masked"] = masked
    return result


def fold_features(train: pd.DataFrame, states: np.ndarray, fold: int):
    max_date = train["time"].max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = train.loc[(train["time"] >= cutoff) & (train["time"] <= max_date)].copy()
    fit = train.loc[train["time"] < cutoff].copy()
    combined_input = pd.concat(
        [fit.drop(columns=["target"]), valid.drop(columns=["target"])],
        ignore_index=True,
        sort=False,
    )
    masked_input = apply_historical_template(combined_input, states, end_at_template_end=True)
    features = core.build_features(masked_input).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    n_fit = len(fit)
    valid_mask = masked_input["TWS_t_masked"].iloc[n_fit:].to_numpy(dtype=bool)
    return (
        features.iloc[:n_fit], fit["target"].to_numpy(dtype=np.float32),
        features.iloc[n_fit:], valid["target"].to_numpy(dtype=np.float32),
        valid_mask, cutoff, max_date,
    )


def model(seed: int, iterations: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=10.0,
        random_strength=1.0,
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


def target_and_prediction(mode: str, features: pd.DataFrame, y: np.ndarray, raw_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = features["TWS_t_ffill"].to_numpy(dtype=np.float32)
    if mode == "delta_ffill":
        return y - baseline, raw_pred + baseline
    return y, raw_pred


def completed() -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if not LOG.exists():
        return done
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "fold":
                done.add((str(row["config"]), int(row["seed"]), int(row["fold"])))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return done


def pair_rows(config: str, seed: int) -> list[dict]:
    if not LOG.exists():
        return []
    rows = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "fold" and row.get("config") == config and int(row.get("seed")) == seed:
                rows.append(row)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def full_features(train: pd.DataFrame, test: pd.DataFrame, states: np.ndarray):
    train_input = apply_historical_template(train.drop(columns=["target"]), states, end_at_template_end=False)
    test_input = test.copy()
    test_input.loc[test_input["TWS_t_masked"].astype(bool), "TWS_t"] = np.nan
    combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
    features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return features.iloc[:len(train)], train["target"].to_numpy(dtype=np.float32), features.iloc[len(train):]


def promote(config: str, mode: str, seed: int, iterations: int, x_train, y_train, x_test, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    train_target, _ = target_and_prediction(mode, x_train, y_train, np.zeros(len(x_train), dtype=np.float32))
    fitted = model(seed, max(100, iterations))
    fitted.fit(x_train, train_target, verbose=False)
    raw = fitted.predict(x_test).astype(np.float32)
    _, pred = target_and_prediction(mode, x_test, np.zeros(len(x_test), dtype=np.float32), raw)
    base = pd.read_csv(SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(dtype=float)
    for weight in (0.2, 0.3, 0.4, 0.5, 0.6, 1.0):
        values = pred if weight == 1.0 else weight * pred + (1.0 - weight) * base
        suffix = "raw" if weight == 1.0 else f"blend{weight:.1f}"
        core.make_submission_frame(test, values, sample).to_csv(
            SUBMISSIONS / f"submission_{config}_seed{seed}_{suffix}.csv", index=False
        )
    write_log({"stage": "full", "config": config, "seed": seed, "mode": mode, "iterations": iterations})
    del fitted
    gc.collect()


def main() -> int:
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    states = template_states(test)
    done = completed()
    prepared = {}
    write_log({"stage": "start", "template": states.astype(int).tolist(), "done": len(done)})
    for config, mode in CONFIGS:
        for seed in SEEDS:
            fold_rows = []
            for fold in (0, 1):
                key = (config, seed, fold)
                if key in done:
                    fold_rows.extend(pair_rows(config, seed))
                    continue
                if fold not in prepared:
                    prepared[fold] = fold_features(train, states, fold)
                x_fit, y_fit, x_valid, y_valid, valid_mask, cutoff, max_date = prepared[fold]
                fit_target, _ = target_and_prediction(mode, x_fit, y_fit, np.zeros(len(x_fit), dtype=np.float32))
                fitted = model(seed, 2200)
                fitted.fit(x_fit, fit_target, eval_set=(x_valid, y_valid if mode == "direct" else y_valid - x_valid["TWS_t_ffill"].to_numpy(dtype=np.float32)), early_stopping_rounds=150, verbose=False)
                iteration = int(fitted.get_best_iteration() + 1)
                raw = fitted.predict(x_valid).astype(np.float32)
                _, prediction = target_and_prediction(mode, x_valid, y_valid, raw)
                baseline = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
                blend = {str(w): core.score(y_valid, w * prediction + (1.0 - w) * baseline, valid_mask) for w in np.arange(0.1, 1.01, 0.1)}
                best_weight = min(blend, key=lambda weight: blend[weight]["all"])
                row = {
                    "stage": "fold", "config": config, "mode": mode, "seed": seed, "fold": fold,
                    "iteration": iteration, "raw": core.score(y_valid, prediction, valid_mask),
                    "best_weight": float(best_weight), "best_blend": blend[best_weight],
                    "period": [str(cutoff.date()), str(max_date.date())],
                }
                write_log(row)
                fold_rows.append(row)
                del fitted
                gc.collect()
            rows = pair_rows(config, seed)
            if len(rows) == 2:
                mean = float(np.mean([row["best_blend"]["all"] for row in rows]))
                iterations = max(100, int(round(np.mean([row["iteration"] for row in rows]))))
                write_log({"stage": "pair", "config": config, "mode": mode, "seed": seed, "mean": mean, "iterations": iterations})
                if not any(row.get("stage") == "full" and row.get("config") == config and int(row.get("seed", -1)) == seed for row in [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines()]):
                    x_train, y_train, x_test = full_features(train, test, states)
                    promote(config, mode, seed, iterations, x_train, y_train, x_test, test, sample)
                    del x_train, y_train, x_test
                    gc.collect()
    write_log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

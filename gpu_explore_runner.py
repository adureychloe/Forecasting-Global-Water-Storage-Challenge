"""Resumable overnight GPU sweep with hyperparameter and spatial variants."""

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
LOG = RESULTS / "explore.jsonl"
PROMOTE_STATE = RESULTS / "explore_promote_state.json"
PROMOTE_THRESHOLD = 0.663713635
FIT_ROWS = 500_000

# The first four configs probe regularisation/randomness around the best d6
# model. The mask variants test the train-time masking prior. Spatial variants
# add same-month cross-sectional statistics from observed inputs only.
CONFIGS = [
    {"name": "cb665_d6_lr03_l2_20", "mask": 0.665, "depth": 6, "lr": 0.03, "l2": 20.0, "rs": 1.0, "spatial": False},
    {"name": "cb665_d6_lr06_l2_5", "mask": 0.665, "depth": 6, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": False},
    {"name": "cb665_d6_l2_30", "mask": 0.665, "depth": 6, "lr": 0.04, "l2": 30.0, "rs": 1.0, "spatial": False},
    {"name": "cb665_d6_rs0", "mask": 0.665, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 0.0, "spatial": False},
    {"name": "cb665_d7", "mask": 0.665, "depth": 7, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb35_d6", "mask": 0.35, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb80_d6", "mask": 0.80, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d6", "mask": 0.90, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb665_d6_spatial", "mask": 0.665, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": True},
    {"name": "cb665_d7_spatial", "mask": 0.665, "depth": 7, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": True},
    {"name": "cb665_d6_lr06_l2_5_spatial", "mask": 0.665, "depth": 6, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": True},
    {"name": "cb90_d6_spatial", "mask": 0.90, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": True},
    {"name": "cb_calendar_d6", "mask": 0.665, "mask_mode": "calendar", "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb_calendar_d6_lr06_l2_5", "mask": 0.665, "mask_mode": "calendar", "depth": 6, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": False},
    {"name": "cb_calendar_d6_spatial", "mask": 0.665, "mask_mode": "calendar", "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": True},
    {"name": "cb90_d6_rows1m", "mask": 0.90, "fit_rows": 1_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb665_d6_lr06_l2_5_spatial_rows1m", "mask": 0.665, "fit_rows": 1_000_000, "depth": 6, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": True},
    {"name": "cb90_d6_rowsall", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d6_lr06_l2_5_rowsall", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d6_lr08_l2_3_rowsall", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.08, "l2": 3.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d7_lr06_l2_5_rowsall", "mask": 0.90, "fit_rows": 5_000_000, "depth": 7, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d8_lr06_l2_5_rowsall", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.06, "l2": 5.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d6_rows1500k", "mask": 0.90, "fit_rows": 1_500_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb80_d6_rowsall", "mask": 0.80, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb95_d6_rowsall", "mask": 0.95, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb100_d6_rowsall", "mask": 1.00, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb_calendar_d6_spatial_rowsall", "mask": 0.665, "mask_mode": "calendar", "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": True},
    {"name": "cb90_d6_rows1750k", "mask": 0.90, "fit_rows": 1_750_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d6_rows2m", "mask": 0.90, "fit_rows": 2_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "spatial": False},
    {"name": "cb90_d6_huber_rowsall", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "loss": "Huber:delta=1", "spatial": False},
    {"name": "cb90_all_bag02", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "bagging_temperature": 0.2, "spatial": False},
    {"name": "cb90_all_bag05", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "bagging_temperature": 0.5, "spatial": False},
    {"name": "cb90_all_bag20", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "bagging_temperature": 2.0, "spatial": False},
    {"name": "cb90_all_bag40", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "bagging_temperature": 4.0, "spatial": False},
    {"name": "cb90_all_rs025", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 0.25, "spatial": False},
    {"name": "cb90_all_rs2", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 2.0, "spatial": False},
    {"name": "cb90_all_leaf5", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "leaf_estimation_iterations": 5, "spatial": False},
    {"name": "cb90_all_leaf10", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "leaf_estimation_iterations": 10, "spatial": False},
    {"name": "cb90_all_bern08", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "bootstrap_type": "Bernoulli", "subsample": 0.8, "spatial": False},
    {"name": "cb90_all_bern095", "mask": 0.90, "fit_rows": 5_000_000, "depth": 6, "lr": 0.04, "l2": 10.0, "rs": 1.0, "bootstrap_type": "Bernoulli", "subsample": 0.95, "spatial": False},
    {"name": "cb90_all_lossguide31", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 31, "spatial": False},
    {"name": "cb90_all_lossguide63", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 63, "spatial": False},
    {"name": "cb90_all_lossguide63_lr06_l2_5", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.06, "l2": 5.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 63, "spatial": False},
    {"name": "cb90_all_lossguide95", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 95, "spatial": False},
    {"name": "cb90_all_lossguide127", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 127, "spatial": False},
    {"name": "cb90_all_lossguide95_lr06_l2_5", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.06, "l2": 5.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 95, "spatial": False},
    {"name": "cb90_all_lossguide127_lr06_l2_5", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.06, "l2": 5.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 127, "spatial": False},
    {"name": "cb90_all_lossguide191", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 191, "spatial": False},
    {"name": "cb90_all_lossguide255", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 255, "spatial": False},
    {"name": "cb90_all_lossguide63_leaf5", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 63, "min_data_in_leaf": 5, "spatial": False},
    {"name": "cb90_all_lossguide63_leaf50", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Lossguide", "max_leaves": 63, "min_data_in_leaf": 50, "spatial": False},
    {"name": "cb90_all_depthwise8", "mask": 0.90, "fit_rows": 5_000_000, "depth": 8, "lr": 0.04, "l2": 10.0, "rs": 1.0, "grow_policy": "Depthwise", "spatial": False},
]
SEEDS = (42, 77, 123, 2027, 9001)

# After the window-alignment audit, prioritize configs that can change the
# leading submission family. Completed v2 promotions remain resumable, while
# these window/loss configs are reached before lower-value control variants.
PRIORITY_NAMES = (
    "cb90_d6_rows1500k",
    "cb90_d6_rows1750k",
    "cb90_d6_rows2m",
    "cb90_d6_rows1m",
    "cb90_d6_huber_rowsall",
    "cb_calendar_d6_spatial_rowsall",
    "cb90_d6_rowsall",
)
# Only these active families need promotion recovery before new folds run.
# Rebuilding every historical promotion here delays unseen configs such as
# rows2m behind dozens of already-synced, lower-value candidates.
RESUME_PROMOTION_NAMES = frozenset((
    "cb90_d6_rows1500k",
    "cb90_d6_rows1750k",
    "cb90_d6_rows2m",
    "cb90_d6_rows1m",
    "cb90_d6_huber_rowsall",
))
CONFIGS = sorted(CONFIGS, key=lambda cfg: (PRIORITY_NAMES.index(cfg["name"]) if cfg["name"] in PRIORITY_NAMES else len(PRIORITY_NAMES), cfg["name"]))


def write_log(row: dict) -> None:
    row = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps(row, ensure_ascii=True), flush=True)


def completed_jobs() -> set[tuple[str, int, int]]:
    done: set[tuple[str, int, int]] = set()
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("stage") == "fold_result":
                    done.add((str(row["config"]), int(row["seed"]), int(row["fold"])))
            except Exception:
                continue
    return done


def promoted_state() -> dict:
    if PROMOTE_STATE.exists():
        try:
            return json.loads(PROMOTE_STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_promoted(state: dict) -> None:
    PROMOTE_STATE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def spatial_build(frame: pd.DataFrame) -> pd.DataFrame:
    """Core leakage-safe features plus contemporaneous input aggregates."""
    result = core.build_features(frame)
    if "time" not in frame.columns:
        return result
    time_key = pd.to_datetime(frame["time"], errors="coerce")
    for col in ("TWS_t", "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t"):
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        grouped = values.groupby(time_key, dropna=False)
        result[f"{col}_time_mean"] = grouped.transform("mean").to_numpy()
        result[f"{col}_time_std"] = grouped.transform("std").fillna(0.0).to_numpy()
    return result


def apply_calendar_mask(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    test_mask = pd.read_csv(core.DATA / "Test.csv", usecols=["time", "TWS_t_masked"])
    test_mask["time"] = pd.to_datetime(test_mask["time"])
    month_rates = test_mask.assign(month=test_mask["time"].dt.month).groupby("month")["TWS_t_masked"].mean()
    row_rates = frame["time"].dt.month.map(month_rates).fillna(float(test_mask["TWS_t_masked"].mean())).to_numpy()
    rng = np.random.default_rng(seed)
    masked = np.zeros(len(frame), dtype=bool)
    for rate in np.unique(row_rates):
        indices = np.flatnonzero(row_rates == rate)
        masked[indices] = True if rate >= 0.9 else (False if rate <= 0.1 else rng.random(len(indices)) < rate)
    frame.loc[masked, "TWS_t"] = np.nan
    frame["TWS_t_masked"] = masked
    return frame


def prepare_fold(frame: pd.DataFrame, fold: int, cfg: dict):
    mask_rate = float(cfg["mask"])
    use_spatial = bool(cfg["spatial"])
    mask_mode = str(cfg.get("mask_mode", "random"))
    core.FIT_MASK_RATE = mask_rate
    max_date = frame["time"].max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[(frame["time"] >= cutoff) & (frame["time"] <= max_date)].copy()
    fit_rows = int(cfg.get("fit_rows", FIT_ROWS))
    fit = frame.loc[frame["time"] < cutoff].tail(fit_rows).copy()
    original_fit = fit.copy() if mask_mode == "calendar" else None
    fit, valid, masked = core.apply_schedule(fit, valid, fold)
    if original_fit is not None:
        fit = apply_calendar_mask(original_fit, seed=42 + fold)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    builder = spatial_build if use_spatial else core.build_features
    features = builder(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return (features.iloc[: len(fit)], fit["target"].to_numpy(dtype=np.float32),
            features.iloc[len(fit) :], valid["target"].to_numpy(dtype=np.float32),
            masked, cutoff, max_date)


def prepare_full(train: pd.DataFrame, test: pd.DataFrame, cfg: dict):
    core.FIT_MASK_RATE = float(cfg["mask"])
    fit_rows = min(int(cfg.get("fit_rows", FIT_ROWS)), len(train))
    fit = train.tail(fit_rows).copy()
    targets = fit["target"].to_numpy(dtype=np.float32)
    train_inputs = fit.drop(columns=["target"]).copy()
    if cfg.get("mask_mode", "random") == "calendar":
        train_inputs = apply_calendar_mask(train_inputs, seed=42)
    else:
        rng = np.random.default_rng(42)
        mask = rng.random(len(train_inputs)) < float(cfg["mask"])
        train_inputs.loc[mask, "TWS_t"] = np.nan
        train_inputs["TWS_t_masked"] = mask
    combined = pd.concat([train_inputs, test.copy()], ignore_index=True, sort=False)
    builder = spatial_build if cfg["spatial"] else core.build_features
    features = builder(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return features.iloc[: len(fit)], targets, features.iloc[len(fit) :]


def score(y: np.ndarray, pred: np.ndarray, masked: np.ndarray) -> dict:
    def r(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    return {"all": r(y, pred), "masked": r(y[masked], pred[masked]), "observed": r(y[~masked], pred[~masked])}


def catboost_options(cfg: dict, seed: int, iterations: int) -> dict:
    options = dict(
        iterations=int(iterations),
        depth=int(cfg["depth"]),
        learning_rate=float(cfg["lr"]),
        l2_leaf_reg=float(cfg["l2"]),
        random_strength=float(cfg["rs"]),
        loss_function=str(cfg.get("loss", "RMSE")),
        eval_metric="RMSE",
        task_type="GPU",
        devices="0",
        random_seed=int(seed),
        border_count=128,
        verbose=False,
        allow_writing_files=False,
    )
    for key in ("bagging_temperature", "bootstrap_type", "subsample", "grow_policy", "max_leaves", "min_data_in_leaf", "leaf_estimation_iterations"):
        if key in cfg:
            options[key] = cfg[key]
    return options


def fit_config(cfg: dict, seed: int, x_fit, y_fit, x_valid, y_valid):
    model = CatBoostRegressor(**catboost_options(cfg, seed, 2200))
    model.fit(x_fit, y_fit, eval_set=(x_valid, y_valid), early_stopping_rounds=150, verbose=False)
    return model, int(model.get_best_iteration() + 1)


def pair_records(config_name: str, seed: int) -> list[dict]:
    rows = []
    if not LOG.exists():
        return rows
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "fold_result" and row.get("config") == config_name and int(row.get("seed")) == seed:
                rows.append(row)
        except Exception:
            continue
    return rows


def pair_mean(config_name: str, seed: int) -> float | None:
    rows = pair_records(config_name, seed)
    if len(rows) != 2:
        return None
    return float(np.mean([r["best_blend"]["all"] for r in rows]))


def promote(
    cfg: dict,
    seed: int,
    iterations: int,
    mean_score: float,
    state: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    full_cache: dict,
) -> None:
    key = f"window-aligned-v2:{cfg['name']}:{seed}"
    if state.get(key, {}).get("status") == "done":
        return
    state[key] = {"status": "running", "mean": mean_score}
    save_promoted(state)
    try:
        cache_key = (
            float(cfg["mask"]), bool(cfg["spatial"]),
            str(cfg.get("mask_mode", "random")), int(cfg.get("fit_rows", FIT_ROWS)),
        )
        if cache_key not in full_cache:
            write_log({"stage": "full_cache_build", "mask_rate": cfg["mask"], "mask_mode": cfg.get("mask_mode", "random"), "spatial": cfg["spatial"]})
            full_cache[cache_key] = prepare_full(train, test, cfg)
        else:
            write_log({"stage": "full_cache_reuse", "mask_rate": cfg["mask"], "mask_mode": cfg.get("mask_mode", "random"), "spatial": cfg["spatial"]})
        x_train, y_train, x_test = full_cache[cache_key]
        model = CatBoostRegressor(**catboost_options(cfg, seed, max(100, int(iterations))))
        model.fit(x_train, y_train, verbose=False)
        pred = model.predict(x_test).astype(float)
        sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
        base_path = SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv"
        base = pd.read_csv(base_path).iloc[:, 1].to_numpy(dtype=float)
        weights = (0.2, 0.3, 0.4, 0.5, 0.6, 1.0)
        for weight in weights:
            values = pred if weight == 1.0 else weight * pred + (1.0 - weight) * base
            suffix = "raw" if weight == 1.0 else f"blend{weight:.1f}"
            out = SUBMISSIONS / f"submission_{cfg['name']}_seed{seed}_{suffix}.csv"
            core.make_submission_frame(test, values, sample).to_csv(out, index=False)
        state[key] = {"status": "done", "mean": mean_score, "iterations": int(iterations)}
        write_log({"stage": "full_candidate", "config": cfg["name"], "seed": seed, "mean": mean_score, "iterations": int(iterations)})
        del model
        gc.collect()
    except Exception as exc:
        state[key] = {"status": "error", "mean": mean_score, "error": repr(exc)}
        write_log({"stage": "full_error", "config": cfg["name"], "seed": seed, "error": repr(exc)})
    save_promoted(state)


def main() -> int:
    os = __import__("os")
    os.chdir(ROOT)
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    done = completed_jobs()
    state = promoted_state()
    prepared = {}
    full_prepared = {}
    write_log({"stage": "start", "configs": [c["name"] for c in CONFIGS], "seeds": list(SEEDS), "done": len(done)})
    # Resume a promotion interrupted by a service/instance restart. Fold
    # results are append-only, so the mean and iteration count are recoverable.
    for cfg in CONFIGS:
        if cfg["name"] not in RESUME_PROMOTION_NAMES:
            continue
        for seed in SEEDS:
            rows = pair_records(cfg["name"], seed)
            if len(rows) != 2:
                continue
            mean_score = float(np.mean([r["best_blend"]["all"] for r in rows]))
            key = f"window-aligned-v2:{cfg['name']}:{seed}"
            if mean_score < PROMOTE_THRESHOLD and state.get(key, {}).get("status") != "done":
                iterations = round(np.mean([int(r["best_iteration"]) for r in rows]))
                write_log({"stage": "resume_promotion", "config": cfg["name"], "seed": seed, "mean": mean_score})
                promote(cfg, seed, iterations, mean_score, state, train, test, full_prepared)
    for cfg in CONFIGS:
        for seed in SEEDS:
            for fold in (0, 1):
                key = (cfg["name"], seed, fold)
                if key in done:
                    continue
                cache_key = (float(cfg["mask"]), bool(cfg["spatial"]), str(cfg.get("mask_mode", "random")), int(cfg.get("fit_rows", FIT_ROWS)), fold)
                if cache_key not in prepared:
                    prepared[cache_key] = prepare_fold(train, fold, cfg)
                x_fit, y_fit, x_valid, y_valid, masked, cutoff, max_date = prepared[cache_key]
                try:
                    model, best_iteration = fit_config(cfg, seed, x_fit, y_fit, x_valid, y_valid)
                    pred = model.predict(x_valid).astype(float)
                    base = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
                    scores = {f"{w:.1f}": score(y_valid, w * pred + (1.0 - w) * base, masked) for w in np.arange(0.1, 1.01, 0.1)}
                    best_weight = min(scores, key=lambda w: scores[w]["all"])
                    write_log({"stage": "fold_result", "config": cfg["name"], "seed": seed, "fold": fold, "mask_rate": cfg["mask"], "mask_mode": cfg.get("mask_mode", "random"), "fit_rows": int(cfg.get("fit_rows", FIT_ROWS)), "loss": cfg.get("loss", "RMSE"), "spatial": cfg["spatial"], "best_iteration": best_iteration, "best_base_blend_weight": float(best_weight), "best_blend": scores[best_weight], "raw": score(y_valid, pred, masked), "period": [str(cutoff.date()), str(max_date.date())]})
                    done.add(key)
                    del model, pred
                    gc.collect()
                    mean_score = pair_mean(cfg["name"], seed)
                    if mean_score is not None and mean_score < PROMOTE_THRESHOLD:
                        pair_rows = pair_records(cfg["name"], seed)
                        iterations = round(np.mean([int(r["best_iteration"]) for r in pair_rows]))
                        promote(cfg, seed, iterations, mean_score, state, train, test, full_prepared)
                except Exception as exc:
                    write_log({"stage": "fold_error", "config": cfg["name"], "seed": seed, "fold": fold, "error": repr(exc)})
    write_log({"stage": "complete", "done": len(done)})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())

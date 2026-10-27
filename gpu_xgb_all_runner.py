"""Resumable all-history mask90 XGBoost GPU sweep.

This route is intentionally separate from the completed CatBoost queue. It
uses the same two deterministic tail folds and leakage-safe feature builder,
then fits the full train history only after a pair of fold results is present.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import gpu_nightly_runner as core


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
LOG = RESULTS / "xgb_all_explore.jsonl"
STATE = RESULTS / "xgb_all_promote_state.json"
THRESHOLD = 0.663713635
SEEDS = (42, 77, 123, 2027, 9001)

CONFIGS = (
    {"name": "xgb90_d6_lr03_l2_20", "depth": 6, "lr": 0.03, "min_child": 20.0, "l2": 20.0},
    {"name": "xgb90_d8_lr025_l2_30", "depth": 8, "lr": 0.025, "min_child": 30.0, "l2": 30.0},
    {"name": "xgb90_d10_lr02_l2_40", "depth": 10, "lr": 0.02, "min_child": 40.0, "l2": 40.0},
)


def log(row: dict) -> None:
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def completed() -> set[tuple[str, int, int]]:
    result: set[tuple[str, int, int]] = set()
    if not LOG.exists():
        return result
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "fold_result":
                result.add((str(row["config"]), int(row["seed"]), int(row["fold"])))
        except Exception:
            continue
    return result


def state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(value: dict) -> None:
    STATE.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def model(cfg: dict, seed: int, estimators: int = 2400, early_stopping: bool = True) -> XGBRegressor:
    params = dict(
        n_estimators=int(estimators),
        max_depth=int(cfg["depth"]),
        learning_rate=float(cfg["lr"]),
        min_child_weight=float(cfg["min_child"]),
        reg_lambda=float(cfg["l2"]),
        reg_alpha=0.1,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=int(seed),
        n_jobs=16,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 150
    return XGBRegressor(**params)


def pair_rows(config: str, seed: int) -> list[dict]:
    rows = []
    if not LOG.exists():
        return rows
    for line in LOG.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "fold_result" and row.get("config") == config and int(row.get("seed")) == seed:
                rows.append(row)
        except Exception:
            continue
    return rows


def promote(cfg: dict, seed: int, iterations: int, mean_score: float, train: pd.DataFrame, test: pd.DataFrame, cache: dict, promotion_state: dict) -> None:
    key = f"{cfg['name']}:{seed}"
    if promotion_state.get(key, {}).get("status") == "done":
        return
    promotion_state[key] = {"status": "running", "mean": mean_score}
    save_state(promotion_state)
    x_train, y_train, x_test = cache.get(("full", seed), (None, None, None))
    if x_train is None:
        train_inputs = train.drop(columns=["target"]).copy()
        mask = np.random.default_rng(int(seed)).random(len(train_inputs)) < core.FIT_MASK_RATE
        train_inputs.loc[mask, "TWS_t"] = np.nan
        train_inputs["TWS_t_masked"] = mask
        combined = pd.concat([train_inputs, test.copy()], ignore_index=True, sort=False)
        features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        x_train, y_train, x_test = features.iloc[: len(train)], train["target"].to_numpy(dtype=np.float32), features.iloc[len(train) :]
        cache[("full", seed)] = (x_train, y_train, x_test)
    fitted = model(cfg, seed, iterations, early_stopping=False)
    fitted.fit(x_train, y_train, verbose=False)
    pred = fitted.predict(x_test).astype(float)
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    base_path = SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv"
    paths = {}
    raw = core.make_submission_frame(test, pred, sample)
    raw_path = SUBMISSIONS / f"submission_{cfg['name']}_seed{seed}_raw.csv"
    raw.to_csv(raw_path, index=False)
    paths["raw"] = str(raw_path)
    if base_path.exists():
        base = pd.read_csv(base_path).iloc[:, 1].to_numpy(float)
        for weight in (0.2, 0.3, 0.4, 0.5, 0.6):
            blended = core.make_submission_frame(test, weight * pred + (1.0 - weight) * base, sample)
            path = SUBMISSIONS / f"submission_{cfg['name']}_seed{seed}_blend{weight:.1f}.csv"
            blended.to_csv(path, index=False)
            paths[f"blend{weight:.1f}"] = str(path)
    promotion_state[key] = {"status": "done", "mean": mean_score, "iterations": iterations, "paths": paths}
    save_state(promotion_state)
    log({"stage": "full_candidate", "config": cfg["name"], "seed": seed, "mean": mean_score, "iterations": iterations})
    del fitted, pred
    gc.collect()


def main() -> int:
    core.FIT_ROWS = 5_000_000
    core.FIT_MASK_RATE = 0.90
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    done = completed()
    promotion_state = state()
    prepared = {}
    full_cache = {}
    log({"stage": "start", "configs": [c["name"] for c in CONFIGS], "seeds": list(SEEDS), "done": len(done)})
    for cfg in CONFIGS:
        for seed in SEEDS:
            pair = pair_rows(cfg["name"], seed)
            if len(pair) != 2:
                continue
            mean_score = float(np.mean([row["best_blend"]["all"] for row in pair]))
            iterations = max(100, int(np.mean([row["best_iteration"] for row in pair])))
            if mean_score < THRESHOLD and promotion_state.get(f"{cfg['name']}:{seed}", {}).get("status") != "done":
                log({"stage": "resume_promotion", "config": cfg["name"], "seed": seed, "mean": mean_score})
                promote(cfg, seed, iterations, mean_score, train, test, full_cache, promotion_state)
    while True:
        made_progress = False
        for cfg in CONFIGS:
            for seed in SEEDS:
                for fold in (0, 1):
                    if (cfg["name"], seed, fold) in done:
                        continue
                    cache_key = (fold,)
                    if cache_key not in prepared:
                        prepared[cache_key] = core.prepare_fold(train, fold)
                    x_fit, y_fit, x_valid, y_valid, masked, cutoff, max_date = prepared[cache_key]
                    fitted = model(cfg, seed)
                    fitted.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)
                    pred = fitted.predict(x_valid).astype(float)
                    base = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
                    scores = {f"{w:.1f}": core.score(y_valid, w * pred + (1.0 - w) * base, masked) for w in np.arange(0.1, 1.01, 0.1)}
                    best_weight = min(scores, key=lambda w: scores[w]["all"])
                    best_iteration = int(fitted.best_iteration + 1)
                    log({"stage": "fold_result", "config": cfg["name"], "seed": seed, "fold": fold, "fit_rows": core.FIT_ROWS, "mask_rate": core.FIT_MASK_RATE, "best_iteration": best_iteration, "best_base_blend_weight": float(best_weight), "best_blend": scores[best_weight], "raw": core.score(y_valid, pred, masked), "period": [str(cutoff.date()), str(max_date.date())]})
                    done.add((cfg["name"], seed, fold))
                    made_progress = True
                    pair = pair_rows(cfg["name"], seed)
                    if len(pair) == 2:
                        mean_score = float(np.mean([row["best_blend"]["all"] for row in pair]))
                        iterations = max(100, int(np.mean([row["best_iteration"] for row in pair])))
                        if mean_score < THRESHOLD:
                            promote(cfg, seed, iterations, mean_score, train, test, full_cache, promotion_state)
                    del fitted, pred
                    gc.collect()
        log({"stage": "cycle_complete", "done": len(done)})
        if len(done) == len(CONFIGS) * len(SEEDS) * 2 and not made_progress:
            time.sleep(60)
        if len(done) == len(CONFIGS) * len(SEEDS) * 2:
            break
    log({"stage": "complete", "done": len(done)})
    while True:
        time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

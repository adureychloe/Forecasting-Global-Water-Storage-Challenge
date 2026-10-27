"""Resumable two-fold GPU model search and full-data candidate generation."""

from __future__ import annotations

import gc
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from baseline import build_features, make_submission_frame


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
RESULTS.mkdir(exist_ok=True)
SUBMISSIONS.mkdir(exist_ok=True)
LEDGER = RESULTS / "gpu_experiments.jsonl"
FIT_ROWS = 500_000
FIT_MASK_RATE = 0.35


CONFIGS = [
    {"name": "cb_d6_lr05_l2_10", "kind": "cat", "iterations": 3000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10.0, "random_strength": 1.0},
    {"name": "cb_d8_lr04_l2_10", "kind": "cat", "iterations": 2800, "depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 10.0, "random_strength": 1.0},
    {"name": "cb_d10_lr03_l2_20", "kind": "cat", "iterations": 2500, "depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 20.0, "random_strength": 0.5},
    {"name": "xgb_d6_lr04", "kind": "xgb", "n_estimators": 3000, "max_depth": 6, "learning_rate": 0.04, "min_child_weight": 10.0, "reg_lambda": 10.0},
    {"name": "xgb_d8_lr03", "kind": "xgb", "n_estimators": 2800, "max_depth": 8, "learning_rate": 0.03, "min_child_weight": 15.0, "reg_lambda": 15.0},
    {"name": "xgb_d10_lr025", "kind": "xgb", "n_estimators": 2400, "max_depth": 10, "learning_rate": 0.025, "min_child_weight": 20.0, "reg_lambda": 20.0},
]


def log(record: dict) -> None:
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(json.dumps(record, ensure_ascii=True), flush=True)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def score(y: np.ndarray, pred: np.ndarray, masked: np.ndarray) -> dict:
    return {"all": rmse(y, pred), "masked": rmse(y[masked], pred[masked]), "observed": rmse(y[~masked], pred[~masked])}


def test_mask_rates() -> np.ndarray:
    test = pd.read_csv(DATA / "Test.csv", usecols=["time", "TWS_t_masked"])
    test["time"] = pd.to_datetime(test["time"])
    return test.groupby("time").TWS_t_masked.mean().sort_index().to_numpy()


def apply_schedule(fit: pd.DataFrame, valid: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42 + fold)
    fit_masked = rng.random(len(fit)) < FIT_MASK_RATE
    rates = test_mask_rates()
    dates = np.sort(valid["time"].unique())
    rate_by_date = dict(zip(dates, rates[-len(dates) :]))
    row_rates = valid["time"].map(rate_by_date).fillna(float(np.mean(rates))).to_numpy()
    valid_masked = np.zeros(len(valid), dtype=bool)
    for rate in np.unique(row_rates):
        idx = np.flatnonzero(row_rates == rate)
        valid_masked[idx] = True if rate >= 0.9 else (False if rate <= 0.1 else rng.random(len(idx)) < rate)
    fit.loc[fit_masked, "TWS_t"] = np.nan
    valid.loc[valid_masked, "TWS_t"] = np.nan
    fit["TWS_t_masked"] = fit_masked
    valid["TWS_t_masked"] = valid_masked
    return fit, valid, valid_masked


def prepare_fold(frame: pd.DataFrame, fold: int):
    max_date = frame["time"].max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[(frame["time"] >= cutoff) & (frame["time"] <= max_date)].copy()
    fit = frame.loc[frame["time"] < cutoff].tail(FIT_ROWS).copy()
    fit, valid, masked = apply_schedule(fit, valid, fold)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    x_fit = features.iloc[: len(fit)]
    x_valid = features.iloc[len(fit) :]
    return x_fit, fit["target"].to_numpy(dtype=np.float32), x_valid, valid["target"].to_numpy(dtype=np.float32), masked, cutoff, max_date


def make_model(config: dict, seed: int = 42, iterations: int | None = None):
    params = {k: v for k, v in config.items() if k not in {"name", "kind"}}
    if iterations is not None:
        params["iterations" if config["kind"] == "cat" else "n_estimators"] = iterations
    if config["kind"] == "cat":
        return CatBoostRegressor(
            **params,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type="GPU",
            devices="0",
            random_seed=seed,
            border_count=128,
            verbose=200,
            allow_writing_files=False,
        )
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        random_state=seed,
        n_jobs=16,
    )


def fit_gpu(config: dict, x_fit, y_fit, x_valid, y_valid):
    model = make_model(config)
    if config["kind"] == "cat":
        model.fit(x_fit, y_fit, eval_set=(x_valid, y_valid), early_stopping_rounds=200, verbose=200)
        best_iteration = int(model.get_best_iteration() + 1)
    else:
        model.set_params(early_stopping_rounds=200)
        model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)
        best_iteration = int(getattr(model, "best_iteration", config["n_estimators"] - 1) + 1)
    return model, best_iteration


def direct_baseline(x_fit, y_fit, x_valid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    ridge.fit(x_fit, y_fit)
    ridge_pred = ridge.predict(x_valid)
    tree = ExtraTreesRegressor(n_estimators=120, max_depth=24, min_samples_leaf=2, max_features=0.8, n_jobs=64, random_state=42)
    tree.fit(x_fit, y_fit)
    tree_pred = tree.predict(x_valid)
    return 0.7 * tree_pred + 0.3 * ridge_pred, tree_pred, ridge_pred


def run_validation(frame: pd.DataFrame) -> dict:
    summary = {}
    for fold in (0, 1):
        cache = RESULTS / f"fold{fold}"
        cache.mkdir(exist_ok=True)
        x_fit, y_fit, x_valid, y_valid, masked, cutoff, max_date = prepare_fold(frame, fold)
        np.save(cache / "y_valid.npy", y_valid)
        np.save(cache / "masked.npy", masked)
        base_path = cache / "direct_base.npy"
        if base_path.exists():
            base_pred = np.load(base_path)
        else:
            base_pred, tree_pred, ridge_pred = direct_baseline(x_fit, y_fit, x_valid)
            np.save(base_path, base_pred)
            np.save(cache / "tree.npy", tree_pred)
            np.save(cache / "ridge.npy", ridge_pred)
            log({"stage": "baseline", "fold": fold, "period": [str(cutoff.date()), str(max_date.date())], "score": score(y_valid, base_pred, masked)})
        summary.setdefault("direct_base", {})[fold] = score(y_valid, base_pred, masked)

        for config in CONFIGS:
            pred_path = cache / f"{config['name']}.npy"
            meta_path = cache / f"{config['name']}.json"
            try:
                if pred_path.exists() and meta_path.exists():
                    pred = np.load(pred_path)
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                else:
                    model, best_iteration = fit_gpu(config, x_fit, y_fit, x_valid, y_valid)
                    pred = model.predict(x_valid).astype(np.float64)
                    meta = {"best_iteration": best_iteration}
                    np.save(pred_path, pred)
                    meta_path.write_text(json.dumps(meta), encoding="utf-8")
                    del model
                    gc.collect()
                raw_score = score(y_valid, pred, masked)
                blend_scores = {}
                for weight in np.arange(0.1, 1.01, 0.1):
                    blended = weight * pred + (1.0 - weight) * base_pred
                    blend_scores[f"{weight:.1f}"] = score(y_valid, blended, masked)
                best_weight = min(blend_scores, key=lambda key: blend_scores[key]["all"])
                result = {"raw": raw_score, "best_iteration": meta["best_iteration"], "best_base_blend_weight": float(best_weight), "best_blend": blend_scores[best_weight]}
                summary.setdefault(config["name"], {})[fold] = result
                log({"stage": "validation", "fold": fold, "model": config["name"], **result})
            except Exception as exc:
                log({"stage": "error", "fold": fold, "model": config["name"], "error": repr(exc), "traceback": traceback.format_exc()})
        del x_fit, y_fit, x_valid, y_valid, masked
        gc.collect()
    (RESULTS / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def rank_configs(summary: dict) -> list[tuple[str, float]]:
    ranked = []
    for config in CONFIGS:
        folds = summary.get(config["name"], {})
        if len(folds) != 2:
            continue
        mean_score = float(np.mean([folds[str(f)]["best_blend"]["all"] if str(f) in folds else folds[f]["best_blend"]["all"] for f in (0, 1)]))
        ranked.append((config["name"], mean_score))
    return sorted(ranked, key=lambda item: item[1])


def prepare_full(train: pd.DataFrame, test: pd.DataFrame):
    targets = train["target"].to_numpy(dtype=np.float32)
    train_inputs = train.drop(columns=["target"]).copy()
    test_inputs = test.copy()
    rng = np.random.default_rng(42)
    mask = rng.random(len(train_inputs)) < FIT_MASK_RATE
    train_inputs.loc[mask, "TWS_t"] = np.nan
    train_inputs["TWS_t_masked"] = mask
    combined = pd.concat([train_inputs, test_inputs], ignore_index=True, sort=False)
    features = build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return features.iloc[: len(train)], targets, features.iloc[len(train) :]


def train_full_candidates(train: pd.DataFrame, test: pd.DataFrame, summary: dict) -> None:
    ranked = rank_configs(summary)
    log({"stage": "ranking", "ranked": ranked})
    if not ranked:
        return
    x_train, y_train, x_test = prepare_full(train, test)
    sample = pd.read_csv(DATA / "SampleSubmission.csv")
    base_path = SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv"
    if not base_path.exists():
        fallback = ROOT / "submission_aug35_ageflag_ens3_ridge30_a100.csv"
        if fallback.exists():
            fallback.replace(base_path)
    base_submission = pd.read_csv(base_path) if base_path.exists() else None

    for model_name, mean_score in ranked[:2]:
        config = next(item for item in CONFIGS if item["name"] == model_name)
        folds = summary[model_name]
        best_iterations = [folds[str(f)]["best_iteration"] if str(f) in folds else folds[f]["best_iteration"] for f in (0, 1)]
        iterations = max(100, int(np.median(best_iterations)))
        try:
            model = make_model(config, iterations=iterations)
            model.fit(x_train, y_train, verbose=200 if config["kind"] == "cat" else False)
            pred = model.predict(x_test).astype(float)
            raw_path = SUBMISSIONS / f"submission_gpu_{model_name}.csv"
            make_submission_frame(test, pred, sample).to_csv(raw_path, index=False)
            if base_submission is not None:
                base_pred = base_submission.iloc[:, 1].to_numpy(dtype=float)
                fold_weights = [folds[str(f)]["best_base_blend_weight"] if str(f) in folds else folds[f]["best_base_blend_weight"] for f in (0, 1)]
                center = float(np.median(fold_weights))
                weights = sorted(set([max(0.1, center - 0.1), center, min(0.9, center + 0.1), 0.2, 0.4, 0.6]))
                for weight in weights:
                    blend = weight * pred + (1.0 - weight) * base_pred
                    path = SUBMISSIONS / f"submission_blend_base{1-weight:.1f}_{model_name}{weight:.1f}.csv"
                    make_submission_frame(test, blend, sample).to_csv(path, index=False)
            log({"stage": "full_candidate", "model": model_name, "mean_fold_rmse": mean_score, "iterations": iterations, "output": str(raw_path)})
            del model
            gc.collect()
        except Exception as exc:
            log({"stage": "full_error", "model": model_name, "error": repr(exc), "traceback": traceback.format_exc()})


def main() -> int:
    os.chdir(ROOT)
    train = pd.read_csv(DATA / "Train.csv")
    test = pd.read_csv(DATA / "Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    log({"stage": "start", "rows_train": len(train), "rows_test": len(test), "configs": [c["name"] for c in CONFIGS]})
    summary = run_validation(train)
    train_full_candidates(train, test, summary)
    log({"stage": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

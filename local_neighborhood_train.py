"""Local CPU/GPU training: drop banned coordinates and add neighbourhood TWS."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from coordinate_compliance import drop_coordinate_features
from enhanced_features import build_features
from neighborhood_tws_features import add_neighborhood_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOG = RESULTS / "local_neighborhood.jsonl"
FIT_ROWS = 500_000


def log(row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def score(y, pred, masked) -> dict:
    return {
        "all": rmse(y, pred),
        "masked": rmse(y[masked], pred[masked]),
        "observed": rmse(y[~masked], pred[~masked]),
    }


def public_states(test: pd.DataFrame) -> np.ndarray:
    rates = test.groupby("time", sort=True)["TWS_t_masked"].mean().to_numpy(dtype=float)
    return rates >= 0.5


def mask_by_template(frame: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    dates = np.sort(result["time"].unique())
    offset = len(dates) - len(states)
    indexes = (np.arange(len(dates)) - offset) % len(states)
    states_by_date = dict(zip(dates, states[indexes]))
    masked = result["time"].map(states_by_date).to_numpy(dtype=bool)
    result.loc[masked, "TWS_t"] = np.nan
    result["TWS_t_masked"] = masked
    return result


def make_lgbm():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def make_hgb():
    return HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        random_state=42,
    )


def make_extra_trees():
    return ExtraTreesRegressor(
        n_estimators=80,
        max_depth=24,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
    )


def available_models():
    models = {"hgb": make_hgb}
    try:
        make_lgbm()
        models = {"lgbm": make_lgbm, **models}
    except Exception as exc:
        log({"stage": "lgbm_unavailable", "error": str(exc)})
    return models


def fold_frame(train: pd.DataFrame, states: np.ndarray, fold: int):
    max_date = train.time.max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = train.loc[(train.time >= cutoff) & (train.time <= max_date)].copy()
    fit = train.loc[train.time < cutoff].tail(FIT_ROWS).copy()
    combined = pd.concat(
        [fit.drop(columns=["target"]), valid.drop(columns=["target"])],
        ignore_index=True,
        sort=False,
    )
    combined = mask_by_template(combined, states)
    return fit, valid, combined, cutoff, max_date


def make_matrices(combined: pd.DataFrame, n_fit: int, with_neighbors: bool):
    features = build_features(combined).replace([np.inf, -np.inf], np.nan)
    if with_neighbors:
        features = add_neighborhood_features(combined, features)
    features = drop_coordinate_features(features).fillna(0.0).astype(np.float32)
    return features.iloc[:n_fit], features.iloc[n_fit:]


def main() -> int:
    train = pd.read_csv(DATA / "Train.csv")
    test = pd.read_csv(DATA / "Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    states = public_states(test)
    models = available_models()
    log(
        {
            "stage": "start",
            "fit_rows": FIT_ROWS,
            "states": states.astype(int).tolist(),
            "models": list(models),
        }
    )
    for fold in (0, 1):
        fit, valid, combined, cutoff, max_date = fold_frame(train, states, fold)
        y_fit = fit["target"].to_numpy(dtype=np.float32)
        y_valid = valid["target"].to_numpy(dtype=np.float32)
        masked = combined["TWS_t_masked"].iloc[len(fit) :].to_numpy(dtype=bool)
        log(
            {
                "stage": "fold_ready",
                "fold": fold,
                "cutoff": str(cutoff.date()),
                "max_date": str(max_date.date()),
                "n_fit": int(len(fit)),
                "n_valid": int(len(valid)),
                "mask_rate": float(masked.mean()),
            }
        )
        for neighbor_flag, tag in ((False, "control"), (True, "neighbor500")):
            started = time.time()
            x_fit, x_valid = make_matrices(combined, len(fit), with_neighbors=neighbor_flag)
            log(
                {
                    "stage": "features",
                    "fold": fold,
                    "tag": tag,
                    "n_features": int(x_fit.shape[1]),
                    "columns": list(x_fit.columns),
                    "seconds": round(time.time() - started, 1),
                }
            )
            for name, factory in models.items():
                model = factory()
                fit_started = time.time()
                model.fit(x_fit, y_fit)
                pred = np.asarray(model.predict(x_valid), dtype=np.float32)
                metrics = score(y_valid, pred, masked)
                log(
                    {
                        "stage": "fold_score",
                        "fold": fold,
                        "tag": tag,
                        "model": name,
                        "metrics": metrics,
                        "fit_seconds": round(time.time() - fit_started, 1),
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

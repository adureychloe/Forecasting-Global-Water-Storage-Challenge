"""Evaluate a small multi-seed clipped ExtraTrees ensemble on the tail protocol."""

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features
from horizon_validation import apply_test_schedule


def score(y, pred, masked):
    return {
        "all": float(np.sqrt(mean_squared_error(y, pred))),
        "masked": float(np.sqrt(mean_squared_error(y[masked], pred[masked]))),
        "observed": float(np.sqrt(mean_squared_error(y[~masked], pred[~masked]))),
    }


def make_tree(seed):
    return ExtraTreesRegressor(n_estimators=100, max_depth=24, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=seed)


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit = frame.loc[frame["time"] < cutoff].tail(500_000).copy()
    fit, valid, fit_masked, valid_masked = apply_test_schedule(fit, valid)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_fit = features.iloc[: len(fit)]
    x_valid = features.iloc[len(fit) :]
    y_fit = fit["target"].to_numpy()
    y_valid = valid["target"].to_numpy()
    clip_rate = float(os.environ.get("CLIP_RATE", "0.02"))
    lo, hi = np.quantile(y_fit, [clip_rate, 1.0 - clip_rate])
    clipped = np.clip(y_fit, lo, hi) if clip_rate > 0.0 else y_fit
    preds = []
    for seed in (7, 42, 99):
        tree = make_tree(seed)
        tree.fit(x_fit, clipped)
        preds.append(tree.predict(x_valid))
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    ridge.fit(x_fit, y_fit)
    ridge_pred = ridge.predict(x_valid)
    outputs = {"single42": score(y_valid, preds[1], valid_masked), "mean3": score(y_valid, np.mean(preds, axis=0), valid_masked)}
    for tree_weight in (0.4, 0.5, 0.6, 0.7, 0.8):
        outputs[f"mean3_ridge_{tree_weight}"] = score(y_valid, tree_weight * np.mean(preds, axis=0) + (1.0 - tree_weight) * ridge_pred, valid_masked)
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

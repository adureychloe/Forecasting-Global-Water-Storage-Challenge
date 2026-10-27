"""Evaluate a HistGradientBoosting model on the deterministic tail mask protocol."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features
from horizon_validation import apply_test_schedule, make_model


def score(y: np.ndarray, pred: np.ndarray, masked: np.ndarray) -> dict:
    return {
        "all": float(np.sqrt(mean_squared_error(y, pred))),
        "masked": float(np.sqrt(mean_squared_error(y[masked], pred[masked]))),
        "observed": float(np.sqrt(mean_squared_error(y[~masked], pred[~masked]))),
    }


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit = frame.loc[frame["time"] < cutoff].tail(500_000).copy()
    fit, valid, _, valid_masked = apply_test_schedule(fit, valid)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_features(combined)
    x_fit = features.iloc[: len(fit)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_valid = features.iloc[len(fit) :].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_fit = fit["target"].to_numpy()
    y_valid = valid["target"].to_numpy()

    hgb = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.05, max_leaf_nodes=63, l2_regularization=1.0, random_state=42)
    hgb.fit(x_fit, y_fit)
    hgb_pred = hgb.predict(x_valid)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    ridge.fit(x_fit, y_fit)
    ridge_pred = ridge.predict(x_valid)
    tree = make_model()
    tree.fit(x_fit, y_fit)
    tree_pred = tree.predict(x_valid)
    outputs = {
        "hgb": score(y_valid, hgb_pred, valid_masked),
        "ridge": score(y_valid, ridge_pred, valid_masked),
        "extra_trees": score(y_valid, tree_pred, valid_masked),
    }
    for hgb_weight in (0.1, 0.2, 0.3, 0.4):
        pred = hgb_weight * hgb_pred + (1.0 - hgb_weight) * (0.6 * tree_pred + 0.4 * ridge_pred)
        outputs[f"hgb_{hgb_weight}_base"] = score(y_valid, pred, valid_masked)
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test a recent-window model ensemble for temporal regime drift."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features
from horizon_validation import apply_test_schedule, make_model


def rmse(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "all": float(np.sqrt(mean_squared_error(y, pred))),
        "masked": float(np.sqrt(mean_squared_error(y[mask], pred[mask]))),
        "observed": float(np.sqrt(mean_squared_error(y[~mask], pred[~mask]))),
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

    tree500 = make_model()
    tree500.fit(x_fit, y_fit)
    tree500_valid = tree500.predict(x_valid)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    ridge.fit(x_fit, y_fit)
    base_valid = 0.6 * tree500_valid + 0.4 * ridge.predict(x_valid)

    window = min(250_000, len(fit))
    tree250 = make_model()
    tree250.fit(x_fit.iloc[-window:], y_fit[-window:])
    recent_valid = tree250.predict(x_valid)
    outputs = {"base500_tree_ridge": rmse(y_valid, base_valid, valid_masked), "recent250_tree": rmse(y_valid, recent_valid, valid_masked)}
    for weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
        outputs[f"recent_weight_{weight}"] = rmse(y_valid, (1.0 - weight) * base_valid + weight * recent_valid, valid_masked)
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

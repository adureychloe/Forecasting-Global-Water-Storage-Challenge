"""Test target clipping as a low-cost robust-loss proxy."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features
from horizon_validation import apply_test_schedule, make_model


def score(y, pred, masked):
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
    features = build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_fit = features.iloc[: len(fit)]
    x_valid = features.iloc[len(fit) :]
    y_fit = fit["target"].to_numpy()
    y_valid = valid["target"].to_numpy()
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    ridge.fit(x_fit, y_fit)
    ridge_valid = ridge.predict(x_valid)
    outputs = {}
    for tail in (0.0025, 0.005, 0.01, 0.02):
        lo, hi = np.quantile(y_fit, [tail, 1.0 - tail])
        clipped = np.clip(y_fit, lo, hi)
        tree = make_model()
        tree.fit(x_fit, clipped)
        tree_valid = tree.predict(x_valid)
        outputs[str(tail)] = {"bounds": [float(lo), float(hi)], "tree": score(y_valid, tree_valid, valid_masked)}
        for weight in (0.4, 0.5, 0.6, 0.7, 0.8):
            outputs[f"{tail}_blend_{weight}"] = score(y_valid, weight * tree_valid + (1.0 - weight) * ridge_valid, valid_masked)
    def result_rmse(item):
        value = item[1]
        return value["all"] if "all" in value else value["tree"]["all"]
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "top": sorted(outputs.items(), key=result_rmse)[:12], "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

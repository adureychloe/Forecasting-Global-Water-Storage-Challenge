"""Leakage-safe seasonal-anomaly target validation under the test mask schedule."""

import numpy as np
import pandas as pd
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


def climatology(source: pd.DataFrame, target: np.ndarray, query: pd.DataFrame) -> np.ndarray:
    keys = [source["lat"], source["lon"], source["time"].dt.month]
    table = pd.DataFrame({"lat": source["lat"].to_numpy(), "lon": source["lon"].to_numpy(), "month": source["time"].dt.month.to_numpy(), "target": target})
    means = table.groupby(["lat", "lon", "month"], sort=False).target.mean()
    global_month = table.groupby("month", sort=False).target.mean().to_dict()
    values = [means.get((lat, lon, month), global_month.get(month, float(np.mean(target)))) for lat, lon, month in zip(query["lat"], query["lon"], query["time"].dt.month)]
    return np.asarray(values, dtype=float)


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit_all = frame.loc[frame["time"] < cutoff].copy()
    fit = fit_all.tail(500_000).copy()
    fit, valid, _, valid_masked = apply_test_schedule(fit, valid)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_features(combined)
    x_fit = features.iloc[: len(fit)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_valid = features.iloc[len(fit) :].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_fit = fit["target"].to_numpy()
    y_valid = valid["target"].to_numpy()

    direct_tree = make_model()
    direct_tree.fit(x_fit, y_fit)
    direct_tree_fit = direct_tree.predict(x_fit)
    direct_tree_valid = direct_tree.predict(x_valid)
    direct_ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    direct_ridge.fit(x_fit, y_fit)
    direct_ridge_fit = direct_ridge.predict(x_fit)
    direct_ridge_valid = direct_ridge.predict(x_valid)
    direct_fit = 0.6 * direct_tree_fit + 0.4 * direct_ridge_fit
    direct_valid = 0.6 * direct_tree_valid + 0.4 * direct_ridge_valid
    outputs = {"direct_base": score(y_valid, direct_valid, valid_masked)}

    scopes = {"tail500k": fit, "all_pre_cutoff": fit_all}
    for scope_name, source in scopes.items():
        source_target = source["target"].to_numpy()
        clim_fit = climatology(source, source_target, fit)
        clim_valid = climatology(source, source_target, valid)
        anomaly_fit = y_fit - clim_fit
        anomaly_tree = make_model()
        anomaly_tree.fit(x_fit, anomaly_fit)
        anomaly_tree_valid = anomaly_tree.predict(x_valid) + clim_valid
        anomaly_ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
        anomaly_ridge.fit(x_fit, anomaly_fit)
        anomaly_ridge_valid = anomaly_ridge.predict(x_valid) + clim_valid
        anomaly_valid = 0.6 * anomaly_tree_valid + 0.4 * anomaly_ridge_valid
        outputs[f"{scope_name}_anomaly_tree"] = score(y_valid, anomaly_tree_valid, valid_masked)
        outputs[f"{scope_name}_anomaly_ridge_blend"] = score(y_valid, anomaly_valid, valid_masked)
        for weight in (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0):
            outputs[f"{scope_name}_anomaly_mix_{weight}"] = score(y_valid, (1.0 - weight) * direct_valid + weight * anomaly_valid, valid_masked)

    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "top": sorted(outputs.items(), key=lambda item: item[1]["all"])[:15], "all": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate leakage-safe spatial/monthly residual corrections on the tail mask protocol."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features
from horizon_validation import apply_test_schedule, make_model


def report(y: np.ndarray, pred: np.ndarray, masked: np.ndarray) -> dict:
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

    tree = make_model()
    tree.fit(x_fit, y_fit)
    tree_fit = tree.predict(x_fit)
    tree_valid = tree.predict(x_valid)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=100.0, solver="lsqr"))
    ridge.fit(x_fit, y_fit)
    ridge_fit = ridge.predict(x_fit)
    ridge_valid = ridge.predict(x_valid)
    base_fit = 0.6 * tree_fit + 0.4 * ridge_fit
    base_valid = 0.6 * tree_valid + 0.4 * ridge_valid

    residual = y_fit - base_fit
    fit_month = fit["time"].dt.month.to_numpy()
    valid_month = valid["time"].dt.month.to_numpy()
    fit_lat = fit["lat"].to_numpy()
    valid_lat = valid["lat"].to_numpy()
    fit_lon = fit["lon"].to_numpy()
    valid_lon = valid["lon"].to_numpy()

    outputs = {"base": report(y_valid, base_valid, valid_masked)}
    for bin_size in (5.0, 10.0, 20.0):
        fit_lat_bin = np.floor(fit_lat / bin_size).astype(int)
        fit_lon_bin = np.floor(fit_lon / bin_size).astype(int)
        valid_lat_bin = np.floor(valid_lat / bin_size).astype(int)
        valid_lon_bin = np.floor(valid_lon / bin_size).astype(int)
        for use_month in (False, True):
            fit_key = list(zip(fit_lat_bin, fit_lon_bin, fit_month if use_month else np.zeros(len(fit), dtype=int)))
            valid_key = list(zip(valid_lat_bin, valid_lon_bin, valid_month if use_month else np.zeros(len(valid), dtype=int)))
            stats = pd.DataFrame({"key": fit_key, "resid": residual}).groupby("key").resid.agg(["mean", "count"])
            correction_map = stats["mean"].to_dict()
            correction = np.array([correction_map.get(key, 0.0) for key in valid_key])
            for shrink in (0.1, 0.25, 0.5, 1.0):
                outputs[f"bin{int(bin_size)}_month{int(use_month)}_shrink{shrink}"] = report(y_valid, base_valid + shrink * correction, valid_masked)

    # A simpler calibration by latitude band and mask status tests whether bias is regime-specific.
    fit_mask = fit["TWS_t_masked"].to_numpy().astype(int)
    valid_mask = valid["TWS_t_masked"].to_numpy().astype(int)
    fit_lat_band = np.floor(fit_lat / 10.0).astype(int)
    valid_lat_band = np.floor(valid_lat / 10.0).astype(int)
    fit_key = list(zip(fit_lat_band, fit_month, fit_mask))
    valid_key = list(zip(valid_lat_band, valid_month, valid_mask))
    stats = pd.DataFrame({"key": fit_key, "resid": residual}).groupby("key").resid.agg(["mean", "count"])
    correction_map = stats["mean"].to_dict()
    correction = np.array([correction_map.get(key, 0.0) for key in valid_key])
    for shrink in (0.1, 0.25, 0.5, 1.0):
        outputs[f"lat10_month_mask_shrink{shrink}"] = report(y_valid, base_valid + shrink * correction, valid_masked)

    ranked = sorted(outputs.items(), key=lambda item: item[1]["all"])
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "top": ranked[:12], "base": outputs["base"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

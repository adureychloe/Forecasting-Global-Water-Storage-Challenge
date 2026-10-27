"""Evaluate leakage-safe TWS-age/mask/month residual calibration."""

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
    age_fit = x_fit["TWS_t_age"].to_numpy()
    age_valid = x_valid["TWS_t_age"].to_numpy()
    age_fit_bin = np.where(age_fit <= 0, 0, np.where(age_fit <= 3, 1, 2))
    age_valid_bin = np.where(age_valid <= 0, 0, np.where(age_valid <= 3, 1, 2))
    fit_month = fit["time"].dt.month.to_numpy()
    valid_month = valid["time"].dt.month.to_numpy()
    fit_mask = fit["TWS_t_masked"].to_numpy().astype(int)
    valid_mask = valid["TWS_t_masked"].to_numpy().astype(int)
    residual = y_fit - base_fit
    outputs = {"base": score(y_valid, base_valid, valid_masked)}
    for include_month in (False, True):
        fit_key = list(zip(age_fit_bin, fit_mask, fit_month if include_month else np.zeros(len(fit), dtype=int)))
        valid_key = list(zip(age_valid_bin, valid_mask, valid_month if include_month else np.zeros(len(valid), dtype=int)))
        correction_map = pd.DataFrame({"key": fit_key, "resid": residual}).groupby("key").resid.mean().to_dict()
        correction = np.array([correction_map.get(key, 0.0) for key in valid_key])
        for shrink in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            outputs[f"month{int(include_month)}_shrink{shrink}"] = score(y_valid, base_valid + shrink * correction, valid_masked)
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "top": sorted(outputs.items(), key=lambda item: item[1]["all"])[:10], "base": outputs["base"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

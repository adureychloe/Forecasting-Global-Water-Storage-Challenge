"""Compare global vs horizon-bucketed delta models on the real mask schedule."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import build_features


def make_model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=80,
        max_depth=24,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
    )


def apply_test_schedule(fit: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    test_mask = pd.read_csv("data/Test.csv", usecols=["time", "TWS_t_masked"])
    test_mask["time"] = pd.to_datetime(test_mask["time"])
    test_rates = test_mask.groupby("time").TWS_t_masked.mean().sort_index().to_numpy()
    fit_masked = rng.random(len(fit)) < 0.35
    valid_dates = np.sort(valid["time"].unique())[-len(test_rates) :]
    date_rates = valid["time"].map(dict(zip(valid_dates, test_rates[-len(valid_dates) :]))).fillna(0.6652631504).to_numpy()
    masked = np.zeros(len(valid), dtype=bool)
    for rate in np.unique(date_rates):
        idx = np.flatnonzero(date_rates == rate)
        masked[idx] = True if rate >= 0.9 else (False if rate <= 0.1 else rng.random(len(idx)) < rate)
    fit.loc[fit_masked, "TWS_t"] = np.nan
    valid.loc[masked, "TWS_t"] = np.nan
    fit["TWS_t_masked"] = fit_masked
    valid["TWS_t_masked"] = masked
    return fit, valid, fit_masked, masked


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit = frame.loc[frame["time"] < cutoff].tail(500_000).copy()
    fit, valid, fit_masked, valid_masked = apply_test_schedule(fit, valid)
    y = valid["target"].to_numpy()
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_features(combined)
    x_fit = features.iloc[: len(fit)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_valid = features.iloc[len(fit) :].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    age_fit = x_fit["TWS_t_age"].to_numpy()
    age_valid = x_valid["TWS_t_age"].to_numpy()
    base_fit = x_fit["TWS_t_ffill"].to_numpy()
    base_valid = x_valid["TWS_t_ffill"].to_numpy()
    target_mode = os.environ.get("TARGET_MODE", "delta")
    train_target = fit["target"].to_numpy() - base_fit if target_mode == "delta" else fit["target"].to_numpy()

    global_model = make_model()
    global_model.fit(x_fit, train_target)
    global_pred = global_model.predict(x_valid)
    if target_mode == "delta":
        global_pred += base_valid

    bucket_pred = np.empty(len(valid), dtype=float)
    buckets = [("observed", age_fit == 0, age_valid == 0), ("short", (age_fit >= 1) & (age_fit <= 3), (age_valid >= 1) & (age_valid <= 3)), ("long", age_fit >= 4, age_valid >= 4)]
    bucket_sizes = {}
    fallback = global_pred
    for name, fit_mask, valid_mask in buckets:
        bucket_sizes[name] = int(fit_mask.sum())
        if fit_mask.sum() < 1000 or valid_mask.sum() == 0:
            bucket_pred[valid_mask] = fallback[valid_mask]
            continue
        model = make_model()
        model.fit(x_fit.iloc[fit_mask], train_target[fit_mask])
        bucket_pred[valid_mask] = model.predict(x_valid.iloc[valid_mask])
        if target_mode == "delta":
            bucket_pred[valid_mask] += base_valid[valid_mask]

    ridge_alphas = [float(value) for value in os.environ.get("RIDGE_ALPHAS", "10").split(",") if value.strip()]
    ridge_predictions = {}
    for alpha in ridge_alphas:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        ridge.fit(x_fit, train_target)
        ridge_pred = ridge.predict(x_valid)
        if target_mode == "delta":
            ridge_pred += base_valid
        ridge_predictions[str(alpha)] = ridge_pred

    def report(name: str, pred: np.ndarray) -> dict:
        return {
            "name": name,
            "rmse_all": float(np.sqrt(mean_squared_error(y, pred))),
            "rmse_masked": float(np.sqrt(mean_squared_error(y[valid_masked], pred[valid_masked]))),
            "rmse_observed": float(np.sqrt(mean_squared_error(y[~valid_masked], pred[~valid_masked]))),
        }

    weights = [float(value) for value in os.environ.get("BLEND_WEIGHTS", "0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0").split(",") if value.strip()]
    blends = {}
    ridge_reports = {}
    for alpha_key, ridge_pred in ridge_predictions.items():
        ridge_reports[alpha_key] = report(f"ridge_{alpha_key}", ridge_pred)
        blends[alpha_key] = {
            str(weight): report(f"extra_trees_{weight}_ridge_{alpha_key}", weight * global_pred + (1.0 - weight) * ridge_pred)
            for weight in weights
        }
    print({"cutoff": str(cutoff.date()), "target_mode": target_mode, "bucket_sizes": bucket_sizes, "mask_rate": float(valid_masked.mean()), "global": report(f"global_{target_mode}", global_pred), "bucketed": report(f"bucketed_{target_mode}", bucket_pred), "ridge": ridge_reports, "ridge_blends": blends})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fast validation that mimics the competition's late-time masked TWS regime."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from baseline import build_features, make_regressor


def main() -> int:
    path = Path("data/Train.csv")
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame["time"])
    frame["time"] = dates
    cutoff = dates.max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[dates >= cutoff].copy()
    fit = frame.loc[dates < cutoff].copy()
    # Keep the most recent fit rows to emphasize the train/test regime.
    fit = fit.tail(500_000).copy()
    rng = np.random.default_rng(42)
    test_mask = pd.read_csv("data/Test.csv", usecols=["time", "TWS_t_masked"])
    test_mask["time"] = pd.to_datetime(test_mask["time"])
    test_rates = test_mask.groupby("time").TWS_t_masked.mean().sort_index().to_numpy()
    fit_mask_rate = float(os.environ.get("FIT_MASK_RATE", "0.35"))
    if os.environ.get("FIT_PATTERN", "0") == "1":
        month_rates = test_mask.assign(month=test_mask.time.dt.month).groupby("month").TWS_t_masked.mean().to_dict()
        fit_date_rates = fit["time"].dt.month.map(month_rates).fillna(fit_mask_rate).to_numpy()
        fit_masked = np.zeros(len(fit), dtype=bool)
        for rate in np.unique(fit_date_rates):
            idx = np.flatnonzero(fit_date_rates == rate)
            fit_masked[idx] = True if rate >= 0.9 else (False if rate <= 0.1 else rng.random(len(idx)) < rate)
    else:
        fit_masked = rng.random(len(fit)) < fit_mask_rate
    valid_dates = np.sort(valid["time"].unique())[-len(test_rates) :]
    rate_by_date = dict(zip(valid_dates, test_rates[-len(valid_dates) :]))
    date_rates = valid["time"].map(rate_by_date).fillna(0.6652631504).to_numpy()
    masked = np.zeros(len(valid), dtype=bool)
    for rate in np.unique(date_rates):
        idx = np.flatnonzero(date_rates == rate)
        if rate >= 0.9:
            masked[idx] = True
        elif rate <= 0.1:
            masked[idx] = False
        else:
            masked[idx] = rng.random(len(idx)) < rate
    fit.loc[fit_masked, "TWS_t"] = np.nan
    valid.loc[masked, "TWS_t"] = np.nan
    fit["TWS_t_masked"] = fit_masked
    valid["TWS_t_masked"] = masked

    y_valid = valid["target"].to_numpy()
    raw_fit = fit.drop(columns=["target"])
    raw_valid = valid.drop(columns=["target"])
    combined = pd.concat([raw_fit, raw_valid], ignore_index=True, sort=False)
    features = build_features(combined)
    x_fit = features.iloc[: len(fit)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_valid = features.iloc[len(fit) :].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = make_regressor("extra_trees")
    model.fit(x_fit, fit["target"].to_numpy())
    pred = model.predict(x_valid)
    persistence = valid["TWS_t"].copy()
    persistence = persistence.groupby([valid["lat"], valid["lon"]], dropna=False).ffill().fillna(0.0)
    scores = {
        "rows_fit": len(fit),
        "rows_valid": len(valid),
        "cutoff": str(cutoff.date()),
        "mask_rate": float(masked.mean()),
        "fit_mask_rate": float(fit_masked.mean()),
        "rmse_all": float(np.sqrt(mean_squared_error(y_valid, pred))),
        "rmse_masked": float(np.sqrt(mean_squared_error(y_valid[masked], pred[masked]))),
        "rmse_observed": float(np.sqrt(mean_squared_error(y_valid[~masked], pred[~masked]))),
        "rmse_ffill_persistence": float(np.sqrt(mean_squared_error(y_valid, persistence))),
    }
    blend_scores = {}
    for weight in (0.4, 0.6, 0.75, 0.9, 1.0):
        blend = weight * pred + (1.0 - weight) * persistence.to_numpy()
        blend_scores[str(weight)] = float(np.sqrt(mean_squared_error(y_valid, blend)))
    scores["blend_rmse"] = blend_scores
    print(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

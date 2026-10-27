"""Test past-only climate accumulation and change features on the tail mask protocol."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from baseline import build_features
from horizon_validation import apply_test_schedule, make_model


def build_climate_features(frame: pd.DataFrame) -> pd.DataFrame:
    base = build_features(frame)
    out = base.copy()
    raw = frame.copy()
    raw["time"] = pd.to_datetime(raw["time"])
    keys = ["lat", "lon"]
    climate_cols = [c for c in raw.columns if "spei" in str(c).lower() or "soil_moisture" in str(c).lower()]
    ordered = raw.sort_values(keys + ["time"], kind="mergesort")
    for col in climate_cols:
        grouped = ordered.groupby(keys, dropna=False)[col]
        values = grouped.transform(lambda s: s.ffill())
        source = values.groupby([ordered[k] for k in keys], dropna=False)
        delta = values - source.shift(1)
        out[f"{col}_delta1"] = delta.reindex(raw.index)
        for window in (3, 6, 12):
            accum = source.transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            out[f"{col}_accum{window}"] = accum.reindex(raw.index)
    return out


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit = frame.loc[frame["time"] < cutoff].tail(500_000).copy()
    fit, valid, _, valid_masked = apply_test_schedule(fit, valid)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_climate_features(combined)
    x_fit = features.iloc[: len(fit)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_valid = features.iloc[len(fit) :].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_fit = fit["target"].to_numpy()
    y_valid = valid["target"].to_numpy()
    model = make_model()
    model.fit(x_fit, y_fit)
    pred = model.predict(x_valid)
    report = {
        "cutoff": str(cutoff.date()),
        "features": int(x_fit.shape[1]),
        "mask_rate": float(valid_masked.mean()),
        "rmse_all": float(np.sqrt(mean_squared_error(y_valid, pred))),
        "rmse_masked": float(np.sqrt(mean_squared_error(y_valid[valid_masked], pred[valid_masked]))),
        "rmse_observed": float(np.sqrt(mean_squared_error(y_valid[~valid_masked], pred[~valid_masked]))),
    }
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

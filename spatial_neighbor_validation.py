"""Evaluate same-date coarse spatial neighbor aggregates without target leakage."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from baseline import build_features
from horizon_validation import apply_test_schedule, make_model


def add_neighbors(raw: pd.DataFrame, features: pd.DataFrame, bin_size: float) -> pd.DataFrame:
    out = features.copy()
    dates = pd.to_datetime(raw["time"])
    tmp = pd.DataFrame({
        "date": dates.to_numpy(),
        "lat_bin": np.floor(raw["lat"].to_numpy() / bin_size).astype(int),
        "lon_bin": np.floor(raw["lon"].to_numpy() / bin_size).astype(int),
    }, index=raw.index)
    keys = ["date", "lat_bin", "lon_bin"]
    for col in ("TWS_t_ffill", "SPEI_01_t", "SPEI_03_t", "SOIL_MOISTURE_t"):
        if col not in features:
            continue
        tmp[col] = features[col].to_numpy()
        stats = tmp.groupby(keys, sort=False)[col].agg(["mean", "std", "count"])
        idx = pd.MultiIndex.from_frame(tmp[keys])
        out[f"neighbor_{col}_mean"] = stats["mean"].reindex(idx).to_numpy()
        out[f"neighbor_{col}_std"] = stats["std"].reindex(idx).fillna(0.0).to_numpy()
        out[f"neighbor_{col}_count"] = stats["count"].reindex(idx).fillna(0.0).to_numpy()
    return out


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit = frame.loc[frame["time"] < cutoff].tail(500_000).copy()
    fit, valid, _, valid_masked = apply_test_schedule(fit, valid)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    base_features = build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    outputs = {}
    y_fit = fit["target"].to_numpy()
    y_valid = valid["target"].to_numpy()
    for bin_size in (5.0, 10.0, 20.0):
        features = add_neighbors(combined, base_features, bin_size)
        x_fit = features.iloc[: len(fit)]
        x_valid = features.iloc[len(fit) :]
        model = make_model()
        model.fit(x_fit, y_fit)
        pred = model.predict(x_valid)
        outputs[str(bin_size)] = {
            "features": int(features.shape[1]),
            "all": float(np.sqrt(mean_squared_error(y_valid, pred))),
            "masked": float(np.sqrt(mean_squared_error(y_valid[valid_masked], pred[valid_masked]))),
            "observed": float(np.sqrt(mean_squared_error(y_valid[~valid_masked], pred[~valid_masked]))),
        }
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

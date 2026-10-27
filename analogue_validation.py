"""Evaluate same-location/month historical state analogues without future leakage."""

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


def make_analogue(source: pd.DataFrame, query: pd.DataFrame, query_features: pd.DataFrame) -> np.ndarray:
    source = source.copy()
    source["month"] = source["time"].dt.month
    query = query.copy()
    query["month"] = query["time"].dt.month
    source_features = build_features(source.drop(columns=["target"], errors="ignore"))
    source_features = source_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    query_features = query_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    state_cols = ["TWS_t_ffill", "SPEI_03_t", "SPEI_12_t", "SOIL_MOISTURE_t"]
    scales = {col: max(float(source_features[col].std()), 1e-3) for col in state_cols if col in source_features and col in query_features}
    source_values = source_features[list(scales)].to_numpy()
    source_target = source["target"].to_numpy()
    groups = {}
    for idx, key in enumerate(zip(source["lat"], source["lon"], source["month"])):
        groups.setdefault(key, []).append(idx)
    analogue = np.empty(len(query), dtype=float)
    global_month = source.groupby(source["month"]).target.mean().to_dict()
    for i, key in enumerate(zip(query["lat"], query["lon"], query["month"])):
        candidates = groups.get(key)
        if not candidates:
            analogue[i] = global_month.get(key[2], float(source["target"].mean()))
            continue
        q = query_features.iloc[i][list(scales)].to_numpy(dtype=float)
        c = source_values[candidates]
        dist = np.nan_to_num((c - q) / np.array([scales[col] for col in scales]), nan=10.0)
        analogue[i] = source_target[candidates[int(np.argmin(np.sum(dist * dist, axis=1)))]]
    return analogue


def main() -> int:
    frame = pd.read_csv("data/Train.csv")
    frame["time"] = pd.to_datetime(frame["time"])
    cutoff = frame["time"].max() - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = frame.loc[frame["time"] >= cutoff].copy()
    fit_all = frame.loc[frame["time"] < cutoff].copy()
    fit = fit_all.tail(500_000).copy()
    fit, valid, _, valid_masked = apply_test_schedule(fit, valid)
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    features = build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_fit = features.iloc[: len(fit)]
    x_valid = features.iloc[len(fit) :]
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
    analogue = make_analogue(fit_all, valid, x_valid)
    outputs = {"base": score(y_valid, base_valid, valid_masked), "analogue": score(y_valid, analogue, valid_masked)}
    for weight in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        outputs[str(weight)] = score(y_valid, (1.0 - weight) * base_valid + weight * analogue, valid_masked)
    print({"cutoff": str(cutoff.date()), "mask_rate": float(valid_masked.mean()), "outputs": outputs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

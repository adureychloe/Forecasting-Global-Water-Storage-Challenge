"""Fast baseline for Zindi Global Water Storage Challenge.

The script intentionally keeps the data contract flexible because Zindi files
are downloaded separately. It accepts CSV/Parquet files in ``data/`` (or the
``WATER_DATA_DIR`` environment variable), infers the date/key/target columns,
and never creates a feature from a future row.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATE_CANDIDATES = ("date", "datetime", "time", "month", "DATE", "Date")
LAT_CANDIDATES = ("lat", "latitude", "LAT", "Latitude")
LON_CANDIDATES = ("lon", "longitude", "LON", "Longitude")
TARGET_HINTS = ("target", "tws_t+1", "tws_next", "tws_target", "label", "y")


def _first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def infer_date_column(frame: pd.DataFrame) -> str | None:
    direct = _first_column(frame.columns, DATE_CANDIDATES)
    if direct:
        return direct
    for column in frame.columns:
        if "date" in str(column).lower() or "time" in str(column).lower():
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return str(column)
    return None


def infer_target_columns(frame: pd.DataFrame, explicit: str | None = None) -> list[str]:
    if explicit:
        requested = [x.strip() for x in explicit.split(",") if x.strip()]
        missing = [x for x in requested if x not in frame.columns]
        if missing:
            raise ValueError(f"Target columns not found: {missing}")
        return requested
    found = []
    for column in frame.columns:
        lower = str(column).lower()
        if any(hint in lower for hint in TARGET_HINTS):
            found.append(str(column))
    if found:
        return found
    # In the released schema the label is commonly the only TWS column whose
    # name contains a future marker; this fallback avoids treating TWS_t as y.
    future_tws = [str(c) for c in frame.columns if "tws" in str(c).lower() and any(x in str(c).lower() for x in ("next", "+1", "target"))]
    if future_tws:
        return future_tws
    raise ValueError("Could not infer target column. Pass --target-col explicitly.")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe numeric features from rows ordered by location/time."""
    result = frame.copy()
    date_col = infer_date_column(result)
    if date_col:
        dates = pd.to_datetime(result[date_col], errors="coerce")
        result["month_sin"] = np.sin(2 * np.pi * dates.dt.month.fillna(1) / 12.0)
        result["month_cos"] = np.cos(2 * np.pi * dates.dt.month.fillna(1) / 12.0)
        result["year_index"] = dates.dt.year.fillna(0).astype(float)

    lat_col = _first_column(result.columns, LAT_CANDIDATES)
    lon_col = _first_column(result.columns, LON_CANDIDATES)
    if lat_col:
        result["lat_sin"] = np.sin(np.deg2rad(pd.to_numeric(result[lat_col], errors="coerce")))
        result["lat_cos"] = np.cos(np.deg2rad(pd.to_numeric(result[lat_col], errors="coerce")))
    if lon_col:
        result["lon_sin"] = np.sin(np.deg2rad(pd.to_numeric(result[lon_col], errors="coerce")))
        result["lon_cos"] = np.cos(np.deg2rad(pd.to_numeric(result[lon_col], errors="coerce")))

    # Lag/rolling features only. Grouping by coordinates prevents a row from
    # one grid cell becoming the history of another cell; no negative shift is used.
    tws_cols = [str(c) for c in result.columns if "tws" in str(c).lower() and "+1" not in str(c).lower() and "next" not in str(c).lower() and "target" not in str(c).lower()]
    spei_cols = [str(c) for c in result.columns if "spei" in str(c).lower()]
    history_cols = tws_cols + spei_cols
    if date_col and history_cols:
        keys = [c for c in (lat_col, lon_col) if c]
        order = keys + [date_col] if keys else [date_col]
        ordered = result.sort_values(order, kind="mergesort")
        for history_col in history_cols:
            if keys:
                grouped = ordered.groupby(keys, dropna=False)[history_col]
                history_values = grouped.transform(lambda s: s.ffill())
                source_grouped = history_values.groupby([ordered[k] for k in keys], dropna=False)
            else:
                history_values = ordered[history_col].ffill()
                source_grouped = None
            if history_col in tws_cols:
                result[f"{history_col}_ffill"] = history_values.reindex(result.index)
                missing = ordered[history_col].isna().astype(float)
                result[f"{history_col}_missing"] = missing.reindex(result.index)
                ordered_dates = pd.to_datetime(ordered[date_col], errors="coerce")
                month_ordinal = pd.Series(
                    ordered_dates.dt.year.to_numpy(dtype=float) * 12.0
                    + ordered_dates.dt.month.to_numpy(dtype=float),
                    index=ordered.index,
                )
                observed_position = month_ordinal.where(missing.eq(0))
                if keys:
                    last_observed = observed_position.groupby([ordered[k] for k in keys], dropna=False).ffill()
                else:
                    last_observed = observed_position.ffill()
                result[f"{history_col}_age"] = (month_ordinal - last_observed).fillna(999.0).reindex(result.index)
            for lag_window in (1, 3, 6, 12):
                lag = source_grouped.shift(lag_window) if source_grouped is not None else history_values.shift(lag_window)
                result[f"{history_col}_lag{lag_window}"] = lag.reindex(result.index)
            for roll_window in (3, 6, 12):
                if source_grouped is not None:
                    rolling = source_grouped.transform(lambda s, w=roll_window: s.shift(1).rolling(w, min_periods=1).mean())
                else:
                    rolling = history_values.shift(1).rolling(roll_window, min_periods=1).mean()
                result[f"{history_col}_roll{roll_window}"] = rolling.reindex(result.index)

    excluded = {date_col, "id", "ID", "target", "Target"}
    numeric = result.select_dtypes(include=[np.number]).copy()
    return numeric.drop(columns=[c for c in excluded if c in numeric.columns], errors="ignore")


def make_time_split(frame: pd.DataFrame, valid_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    date_col = infer_date_column(frame)
    if not date_col:
        order = np.arange(len(frame))
    else:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        order = np.argsort(dates.fillna(pd.Timestamp.min).to_numpy())
    cut = max(1, int(len(order) * (1.0 - valid_fraction)))
    return order[:cut], order[cut:]


def make_regressor(model_name: str = "hgb"):
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=120,
            max_depth=24,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=42,
        )
    if model_name != "hgb":
        raise ValueError(f"Unknown model: {model_name}")
    return HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=31, random_state=42)


def make_submission_frame(test: pd.DataFrame, predictions: np.ndarray, sample: pd.DataFrame | None = None) -> pd.DataFrame:
    id_col = _first_column(test.columns, ("id", "sample_id", "ID"))
    if id_col is None:
        raise ValueError("Test data must contain an ID column for submission")
    pred_frame = pd.DataFrame({"ID": test[id_col].astype(str).to_numpy(), "Target": np.asarray(predictions, dtype=float)})
    if sample is None:
        return pred_frame
    sample_ids = _first_column(sample.columns, ("id", "sample_id", "ID"))
    if sample_ids is None:
        raise ValueError("Sample submission must contain an ID column")
    target_col = "Target" if "Target" in sample.columns else str(sample.columns[-1])
    lookup = pred_frame.set_index("ID")["Target"]
    output = sample.copy()
    output[target_col] = output[sample_ids].astype(str).map(lookup).astype(float)
    return output


def discover_data(data_dir: Path) -> tuple[Path | None, Path | None]:
    files = [p for p in data_dir.glob("*") if p.suffix.lower() in {".csv", ".parquet"}]
    train = next((p for p in files if "train" in p.stem.lower()), None)
    test = next((p for p in files if "test" in p.stem.lower()), None)
    return train, test


def read_table(path: Path, max_rows: int = 0) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if max_rows:
        return pd.read_csv(path, nrows=max_rows)
    return pd.read_csv(path)


def run_baseline(train_path: Path, target_col: str | None = None, max_rows: int = 250_000, model_name: str = "hgb") -> dict:
    frame = read_table(train_path, max_rows=max_rows)
    targets = infer_target_columns(frame, target_col)
    feature_frame = build_features(frame.drop(columns=targets))
    usable = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_idx, valid_idx = make_time_split(frame, valid_fraction=0.2)
    scores = {}
    for target in targets:
        y = pd.to_numeric(frame[target], errors="coerce")
        mask_train = y.iloc[train_idx].notna().to_numpy()
        model = make_regressor(model_name)
        model.fit(usable.iloc[train_idx][mask_train], y.iloc[train_idx][mask_train])
        pred = model.predict(usable.iloc[valid_idx])
        y_valid = y.iloc[valid_idx].to_numpy()
        mask_valid = np.isfinite(y_valid)
        scores[target] = float(np.sqrt(mean_squared_error(y_valid[mask_valid], pred[mask_valid])))
    scores["rmse_mean"] = float(np.mean(list(scores.values())))
    scores["rows"] = int(len(frame))
    scores["targets"] = targets
    return scores


def train_submission(train_path: Path, test_path: Path, sample_path: Path | None, output_path: Path, target_col: str | None, model_name: str, augment_mask_rate: float = 0.0, ridge_blend_weight: float = 1.0, ridge_alpha: float = 10.0, target_clip_rate: float = 0.0, ensemble_seeds: str = "42") -> dict:
    train = read_table(train_path, max_rows=0)
    test = read_table(test_path, max_rows=0)
    targets = infer_target_columns(train, target_col)
    train_inputs = train.drop(columns=targets).copy()
    test_inputs = test.copy()
    if "TWS_t_masked" in test_inputs.columns and "TWS_t_masked" not in train_inputs.columns:
        train_inputs["TWS_t_masked"] = False
    if augment_mask_rate > 0 and "TWS_t" in train_inputs.columns:
        rng = np.random.default_rng(42)
        mask = rng.random(len(train_inputs)) < augment_mask_rate
        train_inputs.loc[mask, "TWS_t"] = np.nan
        if "TWS_t_masked" in train_inputs.columns:
            train_inputs.loc[mask, "TWS_t_masked"] = True
    train_inputs["__is_train"] = True
    test_inputs["__is_train"] = False
    combined = pd.concat([train_inputs, test_inputs], ignore_index=True, sort=False)
    combined_features = build_features(combined.drop(columns=["__is_train"]))
    train_features = combined_features.iloc[: len(train)]
    test_features = combined_features.iloc[len(train) :]
    x_train = train_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_test = test_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(train[targets[0]], errors="coerce")
    mask = y.notna().to_numpy()
    y_fit = y.iloc[mask].to_numpy()
    if target_clip_rate > 0.0:
        lower, upper = np.quantile(y_fit, [target_clip_rate, 1.0 - target_clip_rate])
        y_fit = np.clip(y_fit, lower, upper)
    seed_values = [int(value.strip()) for value in ensemble_seeds.split(",") if value.strip()]
    tree_predictions_list = []
    for seed in seed_values:
        model = make_regressor(model_name)
        if hasattr(model, "random_state"):
            model.set_params(random_state=seed)
        model.fit(x_train.iloc[mask], y_fit)
        tree_predictions_list.append(model.predict(x_test))
    tree_predictions = np.mean(tree_predictions_list, axis=0)
    predictions = tree_predictions
    if ridge_blend_weight < 1.0:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha, solver="lsqr"))
        ridge.fit(x_train.iloc[mask], y.iloc[mask])
        ridge_predictions = ridge.predict(x_test)
        predictions = ridge_blend_weight * tree_predictions + (1.0 - ridge_blend_weight) * ridge_predictions
    predictions = np.clip(predictions, -10_000, 10_000)
    sample = pd.read_csv(sample_path) if sample_path and sample_path.exists() else None
    submission = make_submission_frame(test, predictions, sample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return {"status": "ok", "output": str(output_path), "rows": int(len(submission)), "model": model_name, "augment_mask_rate": augment_mask_rate, "ridge_blend_weight": ridge_blend_weight, "ridge_alpha": ridge_alpha, "target_clip_rate": target_clip_rate, "ensemble_seeds": ensemble_seeds, "targets": targets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("WATER_DATA_DIR", "data"))
    parser.add_argument("--train", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--model", choices=("hgb", "extra_trees"), default="hgb")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--test", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--output", default=str(Path("submission") / "submission_baseline.csv"))
    parser.add_argument("--augment-mask-rate", type=float, default=0.0)
    parser.add_argument("--ridge-blend-weight", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--target-clip-rate", type=float, default=0.0)
    parser.add_argument("--ensemble-seeds", default="42")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    train_path = Path(args.train) if args.train else discover_data(data_dir)[0]
    if train_path is None or not train_path.exists():
        print(json.dumps({"status": "blocked", "reason": "Training data not found", "expected": str(data_dir / "Train.csv")}, ensure_ascii=True))
        return 2
    if args.predict_test:
        test_path = Path(args.test) if args.test else discover_data(data_dir)[1]
        if test_path is None or not test_path.exists():
            print(json.dumps({"status": "blocked", "reason": "Test data not found"}, ensure_ascii=True))
            return 2
        sample_path = Path(args.sample) if args.sample else data_dir / "SampleSubmission.csv"
        result = train_submission(train_path, test_path, sample_path, Path(args.output), args.target_col, args.model, args.augment_mask_rate, args.ridge_blend_weight, args.ridge_alpha, args.target_clip_rate, args.ensemble_seeds)
        print(json.dumps(result, ensure_ascii=True))
        return 0
    result = run_baseline(train_path, args.target_col, args.max_rows, args.model)
    print(json.dumps({"status": "ok", **result}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

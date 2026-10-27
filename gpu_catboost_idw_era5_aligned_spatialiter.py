"""2010+ CatBoost aligned walk with 3 intra-month spatial overlays."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd

import gpu_catboost_idw_era5_aligned as base
import neighborhood_tws_features
from aligned_feedback import walk_aligned
from coordinate_compliance import drop_coordinate_features

INNER = 3


def sequential_test_predict(models, train, train_input, test, y_train):
    n_train = len(train_input)

    def predict_month(history, current):
        combined = pd.concat([history, current], ignore_index=True, sort=False)
        features = (
            base.runner.core.build_features(combined)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
        )
        x_train, x_month = base.runner.augment_supervised_features(
            features.iloc[:n_train].copy(),
            y_train,
            features.iloc[-len(current) :].copy(),
            train,
            current,
        )
        _, bmonth = base.runner.route_groups(x_train, x_month, train, current)
        if "tws_neighbor_anchor_mean" in x_month.columns:
            fallback = x_month["tws_neighbor_anchor_mean"].to_numpy(dtype=np.float32)
        elif "TWS_t_ffill" in x_month.columns:
            fallback = x_month["TWS_t_ffill"].to_numpy(dtype=np.float32)
        else:
            fallback = np.zeros(len(x_month), dtype=np.float32)
        pred_m = base._predict_buckets(models, fallback, x_month, bmonth)
        for _ in range(1, INNER):
            x_month = neighborhood_tws_features.overlay_pred_neighborhood(current, x_month, pred_m)
            x_month = drop_coordinate_features(x_month)
            pred_m = base._predict_buckets(models, fallback, x_month, bmonth)
        gc.collect()
        return pred_m

    def on_month(current, values):
        base.runner.log(
            {
                "stage": "aligned_month",
                "month": str(pd.Timestamp(current.time.iloc[0]).date()),
                "n": int(len(current)),
            }
        )

    return walk_aligned(train_input, test, predict_month, on_month)


base.INNER_ITERS = INNER
base.sequential_test_predict = sequential_test_predict
base.runner.PREFIX = "catboost_horizon_idw_era5_aligned_spatialiter"
base.runner.LOG = base.runner.RESULTS / f"{base.runner.PREFIX}.jsonl"
if __name__ == "__main__":
    raise SystemExit(base.custom_main())

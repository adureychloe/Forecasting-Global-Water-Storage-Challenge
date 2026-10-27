"""Labeled sparse-calendar CV: persist-delta target + aligned walk.

Retrains 68-month #258-protocol heads on y - persist, then walk_aligned
inner_iters=1. Compare pooled RMSE to #262 aligned 0.522228. No Test CSV.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import gpu_sparse_replay as core
from aligned_feedback import walk_aligned
from sparse_replay import calendar_template, make_replay, score_predictions

RUN_ID = "aligned_delta_cv_20260913"
REFERENCE_ALIGNED = 0.522228


def persist_col(features):
    for column in ("TWS_t_ffill", "tws_neighbor_anchor_mean"):
        if column in features.columns:
            return features[column].to_numpy(dtype=np.float32)
    return np.zeros(len(features), dtype=np.float32)


def transform(values, features, variant="raw", inverse=False):
    persist = persist_col(features)
    arr = np.asarray(values, dtype=np.float32)
    return arr + persist if inverse else arr - persist


def rollout(models, train_input, query_input, static_features, folder, log, truth, offsets):
    from sparse_model_features import dynamic_query_features

    cached_columns = [c for c in static_features if str(c).startswith(("era5_", "target_"))]
    static_pred = core.predict_heads(models, static_features, "raw")
    np.save(folder / "static.npy", static_pred)

    def predict_month(history, current):
        month_key = pd.Timestamp(current.time.iloc[0]).strftime("%Y-%m")
        saved = folder / f"month_{month_key}.npy"
        if saved.exists():
            return np.load(saved)
        features = core.clean(dynamic_query_features(history, current))
        features = pd.concat(
            [
                features.drop(columns=cached_columns, errors="ignore"),
                static_features.loc[current.index, cached_columns].reset_index(drop=True),
            ],
            axis=1,
        )
        features = features.reindex(columns=static_features.columns)
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError("query feature schema mismatch")
        pred = core.predict_heads(models, features, "raw")
        np.save(saved, pred)
        return pred

    def on_month(current, pred):
        idx = current.index.to_numpy()
        error = pred.astype(float) - truth[idx]
        mask = current.TWS_t_masked.to_numpy(dtype=bool)
        log(
            stage="month_complete",
            month=str(current.time.iloc[0].date()),
            rows=int(len(current)),
            rmse=float(np.sqrt(np.mean(error ** 2))),
            masked_rmse=float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None,
            offset=int(offsets[idx[0]]),
        )
        gc.collect()

    predictions = walk_aligned(train_input, query_input, predict_month, on_month)
    np.save(folder / "aligned_delta.npy", predictions)
    return static_pred, predictions


def main():
    core.transform = transform
    args = SimpleNamespace(iterations=500, recency_months=68, inner_iters=1, min_head_rows=1000, variant="raw")
    output = core.ROOT / "results" / "aligned_feedback" / RUN_ID
    output.mkdir(parents=True, exist_ok=True)

    def log(**row):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
        text = json.dumps(row, allow_nan=False)
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text, flush=True)

    train = pd.read_csv(core.ROOT / "data" / "Train.csv", parse_dates=["time"])
    template = pd.read_csv(core.ROOT / "data" / "Test.csv", parse_dates=["time"])
    weights = calendar_template(template)["_offset"].value_counts().to_dict()
    states = template.groupby("time", sort=True).TWS_t_masked.mean().ge(0.5).to_numpy()
    anchors = ("2007-09", "2008-06", "2009-01")
    seeds = (2027, 77)
    log(stage="start", recency_months=68, inner_iters=1, reference_aligned=REFERENCE_ALIGNED, transform="persist_delta")
    folds = []
    for anchor in anchors:
        fit, query, truth = make_replay(train, template, anchor)
        prepared = None
        for seed in seeds:
            fold_dir = output / anchor / str(seed)
            fold_dir.mkdir(parents=True, exist_ok=True)
            log(stage="fold_start", anchor=anchor, seed=int(seed), retraining=True)
            if prepared is None:
                prepared = core.prepare(fit, query, states, log)
            train_input, query_input, x_fit, x_query = prepared
            models = core.fit_heads(
                x_fit,
                fit.target.to_numpy(np.float32),
                fit,
                pd.Period(anchor).to_timestamp(),
                seed,
                args,
                fold_dir,
                log,
            )
            static_pred, aligned = rollout(
                models, train_input, query_input, x_query, fold_dir, log, truth, query["_offset"].to_numpy()
            )
            scores = {
                "static": score_predictions(query, truth, static_pred, weights),
                "aligned_delta": score_predictions(query, truth, aligned, weights),
            }
            log(stage="fold_complete", anchor=anchor, seed=int(seed), scores=scores)
            folds.append(
                {
                    "key": f"{anchor}/{seed}",
                    "static": scores["static"]["all"],
                    "aligned_delta": scores["aligned_delta"]["all"],
                    "masked_delta": scores["aligned_delta"]["masked"],
                    "masked_static": scores["static"]["masked"],
                }
            )
            del models
            gc.collect()
        prepared = None
        gc.collect()

    def pooled(name):
        return float(np.mean([fold[name] ** 2 for fold in folds]) ** 0.5)

    report = {
        "complete": True,
        "inner_iters": 1,
        "transform": "persist_delta",
        "mode": "aligned_next_month",
        "folds": folds,
        "pooled": {
            "aligned_delta": pooled("aligned_delta"),
            "static": pooled("static"),
            "reference_aligned": REFERENCE_ALIGNED,
        },
        "full_generated": False,
    }
    core.atomic_json(output / "report.json", report)
    log(stage="complete", pooled=report["pooled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

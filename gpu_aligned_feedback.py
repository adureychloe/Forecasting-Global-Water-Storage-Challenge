"""Reuse #258 heads to score time-aligned feedback (no retraining).

Forecast(t+1) is written into month t+1 when that month is the next query
month. Spatial overlay of t+1 onto month t is disabled (inner_iters=1).
This is the 3-day main-line vs 0.6x: fix recursive semantics, not mix weights.
"""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import gpu_sparse_replay as core
from aligned_feedback import walk_aligned
from sparse_replay import calendar_template, make_replay, score_predictions


def rollout_aligned(models, train_input, query_input, static_features, folder, log, truth, offsets):
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
        entry = {"stage": "month_complete", "month": str(current.time.iloc[0].date()), "rows": int(len(current))}
        if truth is not None:
            idx = current.index.to_numpy()
            error = pred.astype(float) - truth[idx]
            mask = current.TWS_t_masked.to_numpy(dtype=bool)
            entry.update(
                rmse=float(np.sqrt(np.mean(error ** 2))),
                masked_rmse=float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None,
                offset=int(offsets[idx[0]]),
            )
        log(**entry)
        gc.collect()

    predictions = walk_aligned(train_input, query_input, predict_month, on_month)
    np.save(folder / "aligned.npy", predictions)
    return static_pred, predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", default="control_20260910")
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    source = core.ROOT / "results" / "sparse_replay" / args.source_run
    output = core.ROOT / "results" / "aligned_feedback" / args.run_id
    manifest = json.loads((source / "manifest.json").read_text())
    source_report = json.loads((source / "report.json").read_text())
    protocol = manifest["protocol"]
    if not source_report["complete"] or manifest["variant"] != "raw":
        raise ValueError("requires completed raw #258 control")
    for name, expected in manifest["code_sha256"].items():
        if core.digest(core.ROOT / name) != expected:
            raise ValueError(f"source code changed: {name}")
    output.mkdir(parents=True, exist_ok=True)

    def log(**row):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
        text = json.dumps(row, allow_nan=False)
        with (output / "events.jsonl").open("a") as handle:
            handle.write(text + "\n")
        print(text, flush=True)

    from catboost import CatBoostRegressor

    train = pd.read_csv(core.ROOT / "data" / "Train.csv", parse_dates=["time"])
    template = pd.read_csv(core.ROOT / "data" / "Test.csv", parse_dates=["time"])
    weights = calendar_template(template)["_offset"].value_counts().to_dict()
    states = template.groupby("time", sort=True).TWS_t_masked.mean().ge(0.5).to_numpy()
    folds = []
    for anchor in protocol["anchors"]:
        for seed in protocol["seeds"]:
            fold = source / anchor / str(seed)
            model_paths = sorted(fold.glob("*.cbm"))
            if {path.stem for path in model_paths} != set(core.BUCKETS):
                raise ValueError(f"missing heads {anchor}/{seed}")
            models = {path.stem: CatBoostRegressor().load_model(str(path)) for path in model_paths}
            log(stage="fold_start", anchor=anchor, seed=seed, retraining=False)
            fit, query, truth = make_replay(train, template, anchor)
            reference = pd.read_parquet(fold / "predictions.parquet")
            keys = ["time", "lat", "lon", "_offset", "TWS_t_masked"]
            pd.testing.assert_frame_equal(
                query[keys].reset_index(drop=True), reference[keys].reset_index(drop=True)
            )
            train_input, query_input, x_fit, x_query = core.prepare(fit, query, states, log)
            del x_fit
            gc.collect()
            out_fold = output / anchor / str(seed)
            out_fold.mkdir(parents=True, exist_ok=True)
            static_pred, aligned = rollout_aligned(
                models, train_input, query_input, x_query, out_fold, log, truth, query["_offset"].to_numpy()
            )
            scores = {
                "static": score_predictions(query, truth, static_pred, weights),
                "aligned": score_predictions(query, truth, aligned, weights),
                "source_static": score_predictions(query, truth, reference.static.to_numpy(), weights),
                "source_recursive": score_predictions(query, truth, reference.recursive.to_numpy(), weights),
            }
            log(stage="fold_complete", anchor=anchor, seed=int(seed), scores=scores)
            folds.append({"key": f"{anchor}/{seed}", **{k: scores[k]["all"] for k in scores},
                          "masked_aligned": scores["aligned"]["masked"],
                          "masked_static": scores["static"]["masked"]})
            for model in models.values():
                del model
            gc.collect()

    def pooled(name):
        return float(np.mean([fold[name] ** 2 for fold in folds]) ** 0.5)

    report = {
        "complete": True,
        "source_run": args.source_run,
        "inner_iters": 1,
        "mode": "aligned_next_month",
        "folds": folds,
        "pooled": {
            "aligned": pooled("aligned"),
            "static": pooled("static"),
            "source_recursive": pooled("source_recursive"),
            "source_static": pooled("source_static"),
        },
        "full_generated": False,
        "scope": "reuse #258 heads; no Test CSV",
    }
    core.atomic_json(output / "report.json", report)
    log(stage="complete", pooled=report["pooled"], retraining=False)


if __name__ == "__main__":
    main()

"""Labeled sparse-calendar CV: aligned walk with inner_iters=3 on #258 heads.

No 2010+ window exists in Train that contains all 18 Test months (GRACE gaps).
This reuses the #262 protocol (anchors 2007-09/2008-06/2009-01, 68-month heads)
and only changes inference: overlay predicted neighbourhood twice more.
Compare pooled RMSE to #262 aligned 0.522228. No Test CSV, no submit.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import gpu_sparse_replay as core
import neighborhood_tws_features as nb
from aligned_feedback import walk_aligned
from sparse_replay import calendar_template, make_replay, score_predictions

INNER = 3
SOURCE_RUN = "control_20260910"
RUN_ID = "aligned_inner3_20260912"


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
        routed = core.bucket(features)
        for _ in range(1, INNER):
            features = core.clean(nb.overlay_pred_neighborhood(current.reset_index(drop=True), features, pred))
            if not np.array_equal(core.bucket(features), routed):
                raise ValueError("spatial overlay changed age routing")
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
    np.save(folder / "aligned_inner3.npy", predictions)
    return static_pred, predictions


def main():
    source = core.ROOT / "results" / "sparse_replay" / SOURCE_RUN
    reference_path = core.ROOT / "results" / "aligned_feedback" / "aligned_20260910" / "report.json"
    if reference_path.exists():
        reference_report = json.loads(reference_path.read_text())
    else:
        reference_report = {"pooled": {"aligned": 0.522228, "static": 0.526434}}
    output = core.ROOT / "results" / "aligned_feedback" / RUN_ID
    manifest = json.loads((source / "manifest.json").read_text())
    protocol = manifest["protocol"]
    if not json.loads((source / "report.json").read_text())["complete"]:
        raise ValueError("requires completed #258 control")
    output.mkdir(parents=True, exist_ok=True)
    log_path = core.ROOT / "logs" / "gpu-aligned-inner3-cv.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(**row):
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), **row}
        text = json.dumps(row, allow_nan=False)
        with (output / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text, flush=True)

    from catboost import CatBoostRegressor

    train = pd.read_csv(core.ROOT / "data" / "Train.csv", parse_dates=["time"])
    template = pd.read_csv(core.ROOT / "data" / "Test.csv", parse_dates=["time"])
    weights = calendar_template(template)["_offset"].value_counts().to_dict()
    states = template.groupby("time", sort=True).TWS_t_masked.mean().ge(0.5).to_numpy()
    log(stage="start", inner_iters=INNER, source_run=SOURCE_RUN, reference_aligned=reference_report["pooled"]["aligned"])
    folds = []
    for anchor in protocol["anchors"]:
        for seed in protocol["seeds"]:
            fold = source / anchor / str(seed)
            model_paths = sorted(fold.glob("*.cbm"))
            if {path.stem for path in model_paths} != set(core.BUCKETS):
                raise ValueError(f"missing heads {anchor}/{seed}")
            models = {path.stem: CatBoostRegressor().load_model(str(path)) for path in model_paths}
            log(stage="fold_start", anchor=anchor, seed=int(seed), retraining=False)
            fit, query, truth = make_replay(train, template, anchor)
            train_input, query_input, x_fit, x_query = core.prepare(fit, query, states, log)
            del x_fit
            gc.collect()
            out_fold = output / anchor / str(seed)
            out_fold.mkdir(parents=True, exist_ok=True)
            static_pred, aligned = rollout(
                models, train_input, query_input, x_query, out_fold, log, truth, query["_offset"].to_numpy()
            )
            scores = {
                "static": score_predictions(query, truth, static_pred, weights),
                "aligned_inner3": score_predictions(query, truth, aligned, weights),
            }
            log(stage="fold_complete", anchor=anchor, seed=int(seed), scores=scores)
            folds.append(
                {
                    "key": f"{anchor}/{seed}",
                    "static": scores["static"]["all"],
                    "aligned_inner3": scores["aligned_inner3"]["all"],
                    "masked_inner3": scores["aligned_inner3"]["masked"],
                    "masked_static": scores["static"]["masked"],
                }
            )
            for model in models.values():
                del model
            gc.collect()

    def pooled(name):
        return float(np.mean([fold[name] ** 2 for fold in folds]) ** 0.5)

    report = {
        "complete": True,
        "source_run": SOURCE_RUN,
        "inner_iters": INNER,
        "mode": "aligned_next_month",
        "folds": folds,
        "pooled": {
            "aligned_inner3": pooled("aligned_inner3"),
            "static": pooled("static"),
            "reference_aligned_inner1": reference_report["pooled"]["aligned"],
        },
        "full_generated": False,
        "scope": "reuse #258 heads; labeled CV only",
    }
    core.atomic_json(output / "report.json", report)
    log(stage="complete", pooled=report["pooled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

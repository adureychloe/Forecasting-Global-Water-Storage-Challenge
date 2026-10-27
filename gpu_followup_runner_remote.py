"""Second-stage GPU queue: mask-rate and robust-objective sweeps."""

from __future__ import annotations

import gc
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import gpu_nightly_runner as core


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FOLLOW = RESULTS / "followup.jsonl"


def log(record):
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with FOLLOW.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def main() -> int:
    train = pd.read_csv(core.DATA / "Train.csv")
    train["time"] = pd.to_datetime(train["time"])
    configs = [
        ("cb_mask50_d8", "cat", 0.50, "RMSE"),
        ("cb_mask665_d8", "cat", 0.665, "RMSE"),
        ("cb_mask50_huber", "cat", 0.50, "Huber:delta=1"),
        ("cb_mask665_huber", "cat", 0.665, "Huber:delta=1"),
        ("xgb_mask50_pseudohuber", "xgb", 0.50, "reg:pseudohubererror"),
        ("xgb_mask665_pseudohuber", "xgb", 0.665, "reg:pseudohubererror"),
    ]
    prepared = {}
    seeds = (42, 77, 123, 2027, 9001)
    for cycle, seed in zip(itertools.count(), itertools.cycle(seeds)):
      log({"stage": "cycle_start", "cycle": cycle, "seed": seed})
      for name, kind, mask_rate, objective in configs:
        core.FIT_MASK_RATE = mask_rate
        for fold in (0, 1):
            try:
                cache_key = (mask_rate, fold)
                if cache_key not in prepared:
                    prepared[cache_key] = core.prepare_fold(train, fold)
                x_fit, y_fit, x_valid, y_valid, masked, cutoff, max_date = prepared[cache_key]
                base_path = RESULTS / f"fold{fold}" / "direct_base.npy"
                base = np.load(base_path)
                if kind == "cat":
                    model = CatBoostRegressor(iterations=1800, depth=8, learning_rate=0.04, l2_leaf_reg=10.0, loss_function=objective, eval_metric="RMSE", task_type="GPU", devices="0", random_seed=seed, border_count=128, verbose=False, allow_writing_files=False)
                    model.fit(x_fit, y_fit, eval_set=(x_valid, y_valid), early_stopping_rounds=150, verbose=False)
                    best_iteration = int(model.get_best_iteration() + 1)
                else:
                    model = XGBRegressor(n_estimators=1800, max_depth=8, learning_rate=0.03, min_child_weight=15.0, reg_lambda=15.0, objective=objective, eval_metric="rmse", tree_method="hist", device="cuda", max_bin=256, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=16, early_stopping_rounds=150)
                    model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)
                    best_iteration = int(model.best_iteration + 1)
                pred = model.predict(x_valid)
                blend_scores = {}
                for weight in np.arange(0.1, 1.01, 0.1):
                    blend_scores[f"{weight:.1f}"] = core.score(y_valid, weight * pred + (1.0 - weight) * base, masked)
                best_weight = min(blend_scores, key=lambda key: blend_scores[key]["all"])
                log({"stage": "cycle_result", "cycle": cycle, "seed": seed, "model": name, "fold": fold, "mask_rate": mask_rate, "objective": objective, "raw": core.score(y_valid, pred, masked), "best_iteration": best_iteration, "best_base_blend_weight": float(best_weight), "best_blend": blend_scores[best_weight], "period": [str(cutoff.date()), str(max_date.date())]})
                del model, pred
                gc.collect()
            except Exception as exc:
                log({"stage": "cycle_error", "cycle": cycle, "seed": seed, "model": name, "fold": fold, "mask_rate": mask_rate, "objective": objective, "error": repr(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

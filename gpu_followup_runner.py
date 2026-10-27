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


def load_completed() -> set[tuple[int, str, int]]:
    completed = set()
    if not FOLLOW.exists():
        return completed
    for line in FOLLOW.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("stage") == "cycle_result":
                completed.add((int(row["seed"]), str(row["model"]), int(row["fold"])))
        except Exception:
            continue
    return completed


def main() -> int:
    train = pd.read_csv(core.DATA / "Train.csv")
    train["time"] = pd.to_datetime(train["time"])
    # Follow-up route: keep the winning all-history/mask90 data regime fixed and
    # explore CatBoost Bayesian bootstrap temperature, a regularization axis not
    # covered by gpu_explore_runner. Each tuple is (name, family, mask rate,
    # objective, depth, bagging_temperature).
    configs = [
        ("cb90_all_bag02", "cat", 0.90, "RMSE", 6, 0.2),
        ("cb90_all_bag05", "cat", 0.90, "RMSE", 6, 0.5),
        ("cb90_all_bag20", "cat", 0.90, "RMSE", 6, 2.0),
        ("cb90_all_bag40", "cat", 0.90, "RMSE", 6, 4.0),
    ]
    core.FIT_ROWS = 5_000_000
    prepared = {}
    completed = load_completed()
    seeds = (42, 77, 123, 2027, 9001)
    for cycle, seed in zip(itertools.count(), itertools.cycle(seeds)):
      cycle_had_work = False
      log({"stage": "cycle_start", "cycle": cycle, "seed": seed})
      for name, kind, mask_rate, objective, depth, bagging_temperature in configs:
        core.FIT_MASK_RATE = mask_rate
        for fold in (0, 1):
            job_key = (seed, name, fold)
            if job_key in completed:
                continue
            cycle_had_work = True
            try:
                cache_key = (mask_rate, fold)
                if cache_key not in prepared:
                    prepared[cache_key] = core.prepare_fold(train, fold)
                x_fit, y_fit, x_valid, y_valid, masked, cutoff, max_date = prepared[cache_key]
                base_path = RESULTS / f"fold{fold}" / "direct_base.npy"
                base = np.load(base_path)
                if kind == "cat":
                    model = CatBoostRegressor(iterations=2200, depth=depth, learning_rate=0.04, l2_leaf_reg=10.0, bagging_temperature=bagging_temperature, loss_function=objective, eval_metric="RMSE", task_type="GPU", devices="0", random_seed=seed, border_count=128, verbose=False, allow_writing_files=False)
                    model.fit(x_fit, y_fit, eval_set=(x_valid, y_valid), early_stopping_rounds=150, verbose=False)
                    best_iteration = int(model.get_best_iteration() + 1)
                else:
                    model = XGBRegressor(n_estimators=1800, max_depth=depth, learning_rate=0.03, min_child_weight=15.0, reg_lambda=15.0, objective=objective, eval_metric="rmse", tree_method="hist", device="cuda", max_bin=256, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=16, early_stopping_rounds=150)
                    model.fit(x_fit, y_fit, eval_set=[(x_valid, y_valid)], verbose=False)
                    best_iteration = int(model.best_iteration + 1)
                pred = model.predict(x_valid)
                blend_scores = {}
                for weight in np.arange(0.1, 1.01, 0.1):
                    blend_scores[f"{weight:.1f}"] = core.score(y_valid, weight * pred + (1.0 - weight) * base, masked)
                best_weight = min(blend_scores, key=lambda key: blend_scores[key]["all"])
                log({"stage": "cycle_result", "cycle": cycle, "seed": seed, "model": name, "fold": fold, "mask_rate": mask_rate, "objective": objective, "raw": core.score(y_valid, pred, masked), "best_iteration": best_iteration, "best_base_blend_weight": float(best_weight), "best_blend": blend_scores[best_weight], "period": [str(cutoff.date()), str(max_date.date())]})
                completed.add(job_key)
                del model, pred
                gc.collect()
            except Exception as exc:
                log({"stage": "cycle_error", "cycle": cycle, "seed": seed, "model": name, "fold": fold, "mask_rate": mask_rate, "objective": objective, "error": repr(exc)})
      all_done = all((next_seed, config[0], next_fold) in completed for next_seed in seeds for config in configs for next_fold in (0, 1))
      if not cycle_had_work and all_done:
          time.sleep(60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

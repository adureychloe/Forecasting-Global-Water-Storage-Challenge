"""Train one full-data GPU candidate and several base-model blends."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import gpu_nightly_runner as core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--kind", choices=["cat", "xgb"], default="cat")
    parser.add_argument("--objective", default="RMSE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=52)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--name", default="cb_mask50_d8")
    args = parser.parse_args()
    core.FIT_MASK_RATE = args.mask_rate
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    train["time"] = pd.to_datetime(train["time"])
    test["time"] = pd.to_datetime(test["time"])
    x_train, y_train, x_test = core.prepare_full(train, test)
    if args.kind == "cat":
        model = CatBoostRegressor(iterations=args.iterations, depth=args.depth, learning_rate=0.04, l2_leaf_reg=10.0, loss_function=args.objective, task_type="GPU", devices="0", random_seed=args.seed, border_count=128, verbose=200, allow_writing_files=False)
    else:
        model = XGBRegressor(n_estimators=args.iterations, max_depth=args.depth, learning_rate=0.03, min_child_weight=15.0, reg_lambda=15.0, objective=args.objective, eval_metric="rmse", tree_method="hist", device="cuda", max_bin=256, subsample=0.9, colsample_bytree=0.9, random_state=args.seed, n_jobs=16)
    model.fit(x_train, y_train, verbose=200 if args.kind == "cat" else False)
    pred = model.predict(x_test).astype(float)
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    base_path = core.SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv"
    base = pd.read_csv(base_path).iloc[:, 1].to_numpy(dtype=float)
    for weight in (0.2, 0.3, 0.4, 0.5, 0.6, 1.0):
        blended = pred if weight == 1.0 else weight * pred + (1.0 - weight) * base
        suffix = "raw" if weight == 1.0 else f"blend{weight:.1f}"
        out = core.SUBMISSIONS / f"submission_{args.name}_{suffix}.csv"
        core.make_submission_frame(test, blended, sample).to_csv(out, index=False)
        print({"output": str(out), "weight": weight, "mask_rate": args.mask_rate, "iterations": args.iterations}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

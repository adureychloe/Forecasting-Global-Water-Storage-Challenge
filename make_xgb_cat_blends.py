"""Generate low-weight heterogeneous XGBoost/CatBoost candidates."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
SUFFIXES = ("blend0.3", "blend0.4", "blend0.6", "raw")


def avg(paths, out_path, weights):
    frames = [pd.read_csv(p) for p in paths]
    ids = frames[0].iloc[:, 0]
    if any(not ids.equals(f.iloc[:, 0]) for f in frames[1:]):
        raise ValueError("ID order mismatch")
    out = frames[0].copy()
    out.iloc[:, 1] = np.average([f.iloc[:, 1].to_numpy(float) for f in frames], axis=0, weights=weights)
    out.to_csv(out_path, index=False)


def main():
    made = 0
    for suffix in SUFFIXES:
        cat = SUB / f"submission_horizon5_ens5_{suffix}.csv"
        xgb = SUB / f"submission_xgb_horizon5_seed77_{suffix}.csv"
        avg([cat, xgb], SUB / f"submission_horizon5_cat_xgb50_{suffix}.csv", [0.5, 0.5])
        avg([cat, xgb], SUB / f"submission_horizon5_cat_xgb80_{suffix}.csv", [0.8, 0.2])
        avg([cat, xgb], SUB / f"submission_horizon5_cat_xgb90_{suffix}.csv", [0.9, 0.1])
        made += 3
    print({"made": made})


if __name__ == "__main__":
    main()

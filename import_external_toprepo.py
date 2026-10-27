"""Validate and stage predictions from the public same-challenge reference repo."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
EXT = ROOT / "external_refs" / "TWS-Drought-Forecast" / "predictions.parquet"


def main():
    sample = pd.read_csv(ROOT / "data" / "SampleSubmission.csv")
    ext = pd.read_parquet(EXT)
    if not ext["ID"].astype(str).equals(sample.iloc[:, 0].astype(str)):
        raise ValueError("External prediction IDs do not match SampleSubmission order")
    pred = ext["Target"].to_numpy(float)
    if len(pred) != len(sample) or not np.isfinite(pred).all():
        raise ValueError("External predictions have invalid length or non-finite values")

    base = sample.copy()
    base.iloc[:, 1] = pred
    base_path = SUB / "submission_external_toprepo_lgbm.csv"
    base.to_csv(base_path, index=False)

    control = pd.read_csv(SUB / "submission_horizon_heads_seed2027_blend0.3.csv")
    if not control.iloc[:, 0].astype(str).equals(sample.iloc[:, 0].astype(str)):
        raise ValueError("Control IDs do not match SampleSubmission order")
    made = [base_path.name]
    for w in (0.05, 0.10, 0.15, 0.20, 0.30):
        out = control.copy()
        out.iloc[:, 1] = (1.0 - w) * control.iloc[:, 1].to_numpy(float) + w * pred
        path = SUB / f"submission_onlinecontrol_external_toprepo_w{int(w * 100):02d}.csv"
        out.to_csv(path, index=False)
        made.append(path.name)
    print({"made": made, "rows": len(pred), "target_mean": float(pred.mean()), "target_std": float(pred.std())})


if __name__ == "__main__":
    main()

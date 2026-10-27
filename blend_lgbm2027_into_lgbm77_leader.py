"""Blend LGBM seed2027 into the new 0.7143 LGBM seed77 2.5% leader."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_lgbmpattern_seed77_raw_w025.csv"
SRC = SUB / "submission_lgbm_horizon_idw_era5_pattern_seed2027_lgbmpattern_raw.csv"
WEIGHTS = (0.025, 0.05, 0.10)


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = pd.read_csv(LEADER)
    src = pd.read_csv(SRC)
    if not leader.iloc[:, 0].equals(ids) or not src.iloc[:, 0].equals(ids):
        raise ValueError("ID mismatch")
    lv = leader.iloc[:, 1].to_numpy(np.float64)
    sv = src.iloc[:, 1].to_numpy(np.float64)
    if not np.isfinite(lv).all() or not np.isfinite(sv).all():
        raise ValueError("non-finite")
    wrote = []
    for weight in WEIGHTS:
        out = leader.copy()
        mixed = (1.0 - weight) * lv + weight * sv
        out.iloc[:, 1] = mixed
        name = f"submission_onlineleader_idw_era5_lgbmpattern_seed2027_raw_w{int(weight * 1000):03d}.csv"
        path = SUB / name
        out.to_csv(path, index=False)
        wrote.append(name)
    print({"leader": LEADER.name, "rows": int(len(ids)), "wrote": wrote})


if __name__ == "__main__":
    main()

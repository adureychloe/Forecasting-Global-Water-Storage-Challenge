"""Blend LightGBM sources into the new 0.7147 extra-XGB ens2 leader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_xgbextra_ens2_raw_w025.csv"
WEIGHTS = (0.025, 0.05, 0.10)


def load_pred(path: Path, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != len(ids):
        raise ValueError(f"{path.name}: {len(frame)} rows, expected {len(ids)}")
    if not frame.iloc[:, 0].equals(ids):
        raise ValueError(f"{path.name}: ID mismatch")
    values = frame.iloc[:, 1].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: non-finite predictions")
    return values


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load_pred(LEADER, ids)
    wrote = []
    for seed in (77, 2027):
        src = load_pred(SUB / f"submission_lgbm_horizon_idw_era5_pattern_seed{seed}_lgbmpattern_raw.csv", ids)
        for weight in WEIGHTS:
            mixed = (1.0 - weight) * leader + weight * src
            name = f"submission_onlineleader_idw_era5_lgbmpattern_seed{seed}_raw_w{int(weight * 1000):03d}.csv"
            out = pd.read_csv(LEADER)
            out.iloc[:, 1] = mixed
            path = SUB / name
            out.to_csv(path, index=False)
            check = load_pred(path, ids)
            if np.max(np.abs(check - mixed)) > 1e-12:
                raise ValueError(f"rewrite drift {name}")
            wrote.append(name)
    print({"leader": LEADER.name, "rows": int(len(ids)), "wrote": wrote})


if __name__ == "__main__":
    main()

"""Blend XGB lagged-ERA5-anomaly sources into the 0.7139 LGBM seed2027 leader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_lgbmpattern_seed2027_raw_w025.csv"
WEIGHTS = (0.025, 0.05, 0.10)
SEEDS = (77, 2027)


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


def write_mix(ids: pd.Series, mixed: np.ndarray, name: str) -> str:
    out = pd.read_csv(LEADER)
    out.iloc[:, 1] = mixed
    path = SUB / name
    out.to_csv(path, index=False)
    check = load_pred(path, ids)
    if np.max(np.abs(check - mixed)) > 1e-12:
        raise ValueError(f"rewrite drift {name}")
    return name


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load_pred(LEADER, ids)
    wrote = []
    srcs = {}
    for seed in SEEDS:
        src_path = SUB / f"submission_xgb_horizon_idw_era5_anomalag_seed{seed}_xgbanomalag_raw.csv"
        if not src_path.exists():
            continue
        srcs[seed] = load_pred(src_path, ids)
        for weight in WEIGHTS:
            mixed = (1.0 - weight) * leader + weight * srcs[seed]
            name = f"submission_onlineleader_idw_era5_xgbanomalag_seed{seed}_raw_w{int(weight * 1000):03d}.csv"
            wrote.append(write_mix(ids, mixed, name))
    if len(srcs) == 2:
        ens = 0.5 * srcs[77] + 0.5 * srcs[2027]
        mixed = 0.975 * leader + 0.025 * ens
        wrote.append(write_mix(ids, mixed, "submission_onlineleader_idw_era5_xgbanomalag_ens2_raw_w025.csv"))
    if not wrote:
        raise SystemExit("no xgbanomalag sources found")
    print({"leader": LEADER.name, "rows": int(len(ids)), "wrote": wrote})


if __name__ == "__main__":
    main()

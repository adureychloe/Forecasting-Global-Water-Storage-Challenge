"""Blend remaining sources into the new 0.715 XGB 10% leader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_xgbpattern_seed77_raw_w100.csv"
WEIGHTS = (0.025, 0.05, 0.10)
SOURCES = (
    ("xgbpattern", "submission_xgb_horizon_idw_era5_pattern_seed2027_xgbpattern_raw.csv", (2027,)),
    ("patternextra", "submission_catboost_horizon_idw_era5_pattern_extra_seed{seed}_patternextra_raw.csv", (123, 9001)),
)


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


def write_blend(ids, leader, src, name):
    out = pd.read_csv(LEADER)
    out.iloc[:, 1] = src
    path = SUB / name
    out.to_csv(path, index=False)
    check = load_pred(path, ids)
    if np.max(np.abs(check - src)) > 1e-12:
        raise ValueError(f"rewrite drift {name}")
    return name


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load_pred(LEADER, ids)
    wrote = []
    extras = {}
    for family, template, seeds in SOURCES:
        for seed in seeds:
            path = SUB / template.format(seed=seed)
            src = load_pred(path, ids)
            if family == "patternextra":
                extras[seed] = src
            for weight in WEIGHTS:
                mixed = (1.0 - weight) * leader + weight * src
                name = f"submission_onlineleader_idw_era5_{family}_seed{seed}_raw_w{int(weight * 1000):03d}.csv"
                wrote.append(write_blend(ids, leader, mixed, name))
    if extras:
        ens2 = 0.5 * extras[123] + 0.5 * extras[9001]
        p77 = load_pred(SUB / "submission_catboost_horizon_idw_era5_pattern_seed77_pattern_raw.csv", ids)
        p2027 = load_pred(SUB / "submission_catboost_horizon_idw_era5_pattern_seed2027_pattern_raw.csv", ids)
        ens4 = 0.25 * (extras[123] + extras[9001] + p77 + p2027)
        for label, arr in (("ens2", ens2), ("ens4", ens4)):
            mixed = 0.975 * leader + 0.025 * arr
            wrote.append(write_blend(ids, leader, mixed, f"submission_onlineleader_idw_era5_patternextra_{label}_raw_w025.csv"))
    print({"leader": LEADER.name, "rows": int(len(ids)), "wrote": wrote})


if __name__ == "__main__":
    main()

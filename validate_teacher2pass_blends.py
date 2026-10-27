"""Validate #238 sources and write teacher2pass_raw blends vs the 0.7179 leader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_spatialiter_seed77_raw_w100.csv"
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


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    leader = load_pred(LEADER, ids)
    report = {"rows": int(len(ids)), "leader": LEADER.name, "compare": {}, "wrote": []}
    for seed in (77, 2027):
        teacher = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_teacher2pass_seed{seed}_teacher2pass_raw.csv",
            ids,
        )
        static = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_teacher2pass_seed{seed}_static_raw.csv",
            ids,
        )
        pattern = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_pattern_seed{seed}_pattern_raw.csv",
            ids,
        )
        recency = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_recency2012_seed{seed}_recency2012_raw.csv",
            ids,
        )
        spatial = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_spatialiter_seed{seed}_spatialiter_raw.csv",
            ids,
        )
        weighted = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_weighted_seed{seed}_weighted_raw.csv",
            ids,
        )
        report["compare"][seed] = {
            "vs_static": {"rms": rms(teacher, static), "corr": corr(teacher, static)},
            "vs_pattern": {"rms": rms(teacher, pattern), "corr": corr(teacher, pattern)},
            "vs_recency2012": {"rms": rms(teacher, recency), "corr": corr(teacher, recency)},
            "vs_spatialiter": {"rms": rms(teacher, spatial), "corr": corr(teacher, spatial)},
            "vs_weighted": {"rms": rms(teacher, weighted), "corr": corr(teacher, weighted)},
        }
        for weight in WEIGHTS:
            out = pd.read_csv(LEADER)
            out.iloc[:, 1] = (1.0 - weight) * leader + weight * teacher
            values = out.iloc[:, 1].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite blend seed{seed} w{weight}")
            name = f"submission_onlineleader_idw_era5_teacher2pass_seed{seed}_raw_w{int(weight * 1000):03d}.csv"
            path = SUB / name
            out.to_csv(path, index=False)
            check = load_pred(path, ids)
            if np.max(np.abs(check - values)) > 1e-12:
                raise ValueError(f"rewrite drift {name}")
            report["wrote"].append(name)
    print(report)


if __name__ == "__main__":
    main()

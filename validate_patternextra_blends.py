"""Validate #243 extra-seed sources and write patternextra_raw blends vs 0.7173."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"
SAMPLE = ROOT / "data" / "SampleSubmission.csv"
LEADER = SUB / "submission_onlineleader_idw_era5_pattern_seed77_raw_w100.csv"
WEIGHTS = (0.025, 0.05, 0.10)
EXTRA_SEEDS = (123, 9001)
PATTERN_SEEDS = (77, 2027)


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


def write_blend(leader_frame: pd.DataFrame, leader: np.ndarray, src: np.ndarray, weight: float, name: str, ids: pd.Series, wrote: list[str]) -> None:
    out = leader_frame.copy()
    out.iloc[:, 1] = (1.0 - weight) * leader + weight * src
    values = out.iloc[:, 1].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite blend {name}")
    path = SUB / name
    out.to_csv(path, index=False)
    check = load_pred(path, ids)
    if np.max(np.abs(check - values)) > 1e-12:
        raise ValueError(f"rewrite drift {name}")
    wrote.append(name)


def write_source(ids: pd.Series, values: np.ndarray, name: str, wrote: list[str]) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite source {name}")
    out = pd.read_csv(SAMPLE)
    out.iloc[:, 1] = values
    path = SUB / name
    out.to_csv(path, index=False)
    check = load_pred(path, ids)
    if np.max(np.abs(check - values)) > 1e-12:
        raise ValueError(f"rewrite drift {name}")
    wrote.append(name)


def main() -> None:
    ids = pd.read_csv(SAMPLE).iloc[:, 0]
    if len(ids) != 280961:
        raise ValueError(f"sample rows {len(ids)}")
    leader_frame = pd.read_csv(LEADER)
    leader = load_pred(LEADER, ids)
    pattern77 = load_pred(SUB / "submission_catboost_horizon_idw_era5_pattern_seed77_pattern_raw.csv", ids)
    extras = {}
    statics = {}
    report = {"rows": int(len(ids)), "leader": LEADER.name, "compare": {}, "wrote": []}
    for seed in EXTRA_SEEDS:
        extra = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_pattern_extra_seed{seed}_patternextra_raw.csv",
            ids,
        )
        static = load_pred(
            SUB / f"submission_catboost_horizon_idw_era5_pattern_extra_seed{seed}_static_raw.csv",
            ids,
        )
        extras[seed] = extra
        statics[seed] = static
        report["compare"][seed] = {
            "vs_pattern77": {"rms": rms(extra, pattern77), "corr": corr(extra, pattern77)},
            "vs_static": {"rms": rms(extra, static), "corr": corr(extra, static)},
        }
        for weight in WEIGHTS:
            write_blend(
                leader_frame,
                leader,
                extra,
                weight,
                f"submission_onlineleader_idw_era5_patternextra_seed{seed}_raw_w{int(weight * 1000):03d}.csv",
                ids,
                report["wrote"],
            )

    ens2 = 0.5 * extras[123] + 0.5 * extras[9001]
    write_source(
        ids,
        ens2,
        "submission_catboost_horizon_idw_era5_pattern_extra_ens2_patternextra_raw.csv",
        report["wrote"],
    )
    report["compare"]["ens2"] = {
        "vs_pattern77": {"rms": rms(ens2, pattern77), "corr": corr(ens2, pattern77)},
        "vs_123": {"rms": rms(ens2, extras[123]), "corr": corr(ens2, extras[123])},
        "vs_9001": {"rms": rms(ens2, extras[9001]), "corr": corr(ens2, extras[9001])},
    }
    write_blend(
        leader_frame,
        leader,
        ens2,
        0.025,
        "submission_onlineleader_idw_era5_patternextra_ens2_raw_w025.csv",
        ids,
        report["wrote"],
    )

    patterns = {77: pattern77}
    for seed in PATTERN_SEEDS:
        if seed == 77:
            continue
        path = SUB / f"submission_catboost_horizon_idw_era5_pattern_seed{seed}_pattern_raw.csv"
        if path.exists():
            patterns[seed] = load_pred(path, ids)
    if all(seed in patterns for seed in PATTERN_SEEDS):
        ens4 = 0.25 * (patterns[77] + patterns[2027] + extras[123] + extras[9001])
        write_source(
            ids,
            ens4,
            "submission_catboost_horizon_idw_era5_pattern_extra_ens4_patternextra_raw.csv",
            report["wrote"],
        )
        report["compare"]["ens4"] = {
            "vs_pattern77": {"rms": rms(ens4, pattern77), "corr": corr(ens4, pattern77)},
        }
        write_blend(
            leader_frame,
            leader,
            ens4,
            0.025,
            "submission_onlineleader_idw_era5_patternextra_ens4_raw_w025.csv",
            ids,
            report["wrote"],
        )
    report["compare"]["123_vs_9001"] = {
        "rms": rms(extras[123], extras[9001]),
        "corr": corr(extras[123], extras[9001]),
    }
    print(report)


if __name__ == "__main__":
    main()

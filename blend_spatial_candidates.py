"""Build candidate-only ensembles from the strongest spatial and mask90 models."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
SUFFIXES = ("blend0.2", "blend0.3", "blend0.4", "blend0.5", "blend0.6", "raw")


def average(paths: list[Path], output: Path, weights: list[float] | None = None) -> None:
    frames = [pd.read_csv(path) for path in paths]
    if any(not frames[0].iloc[:, 0].equals(frame.iloc[:, 0]) for frame in frames[1:]):
        raise ValueError(f"ID order mismatch for {output.name}")
    result = frames[0].copy()
    values = np.stack([frame.iloc[:, 1].to_numpy(float) for frame in frames])
    result.iloc[:, 1] = np.average(values, axis=0, weights=weights)
    result.to_csv(output, index=False)


def main() -> int:
    made = 0
    for suffix in SUFFIXES:
        spatial = [SUB / f"submission_cb665_d6_spatial_seed{seed}_{suffix}.csv" for seed in (42, 2027, 9001)]
        mask90 = [SUB / f"submission_cb90_d6_seed{seed}_{suffix}.csv" for seed in (42, 123)]
        average(spatial, SUB / f"submission_cb665_d6_spatial_ens3_{suffix}.csv")
        average(mask90, SUB / f"submission_cb90_d6_ens42_123_{suffix}.csv")
        average(
            [spatial[0], *mask90],
            SUB / f"submission_cb665_spatial42_mask90ens2_{suffix}.csv",
            weights=[0.5, 0.25, 0.25],
        )
        calendar = [SUB / f"submission_cb_calendar_d6_spatial_seed{seed}_{suffix}.csv" for seed in (123, 9001)]
        average(calendar, SUB / f"submission_cb_calendar_d6_spatial_ens123_9001_{suffix}.csv")
        average(
            [SUB / f"submission_cb665_d6_lr06_l2_5_spatial_seed42_{suffix}.csv", *calendar],
            SUB / f"submission_cb_spatial_lr06_42_calendar_ens2_{suffix}.csv",
            weights=[0.5, 0.25, 0.25],
        )
        rows1m = [SUB / f"submission_cb90_d6_rows1m_seed{seed}_{suffix}.csv" for seed in (42, 77, 2027, 9001)]
        average(rows1m, SUB / f"submission_cb90_d6_rows1m_ens4_{suffix}.csv")
        average(
            [rows1m[0], *calendar],
            SUB / f"submission_cb90_rows1m42_calendar_ens2_{suffix}.csv",
            weights=[0.5, 0.25, 0.25],
        )
        main_paths = [
            SUB / f"submission_cb90_d6_rows1500k_seed2027_{suffix}.csv",
            SUB / f"submission_cb90_d6_rowsall_seed77_{suffix}.csv",
            SUB / f"submission_cb90_d7_lr06_l2_5_rowsall_seed123_{suffix}.csv",
        ]
        average(main_paths, SUB / f"submission_main3_rows1500k_allhist_d7_{suffix}.csv")
        calendar_rowsall = SUB / f"submission_cb_calendar_d6_spatial_rowsall_seed2027_{suffix}.csv"
        average(
            [*main_paths, calendar_rowsall],
            SUB / f"submission_main4_rows1500k_allhist_d7_calendar_{suffix}.csv",
        )
        average(
            [*main_paths, calendar_rowsall],
            SUB / f"submission_main4_rows1500k40_allhist20_d720_calendar20_{suffix}.csv",
            weights=[0.4, 0.2, 0.2, 0.2],
        )
        made += 10
    print({"made": made})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

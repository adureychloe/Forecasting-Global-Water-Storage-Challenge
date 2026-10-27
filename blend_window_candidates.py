"""Build candidate-only blends across corrected recency windows."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
SUFFIXES = ("blend0.2", "blend0.3", "blend0.4", "blend0.5", "blend0.6", "raw")


def average(names: list[str], output: str, weights: list[float] | None = None) -> None:
    frames = [pd.read_csv(SUB / name) for name in names]
    ids = frames[0].iloc[:, 0]
    if any(not ids.equals(frame.iloc[:, 0]) for frame in frames[1:]):
        raise ValueError(f"ID order mismatch for {output}")
    result = frames[0].copy()
    values = np.stack([frame.iloc[:, 1].to_numpy(float) for frame in frames])
    result.iloc[:, 1] = np.average(values, axis=0, weights=weights)
    result.to_csv(SUB / output, index=False)


def main() -> int:
    made = 0
    for suffix in SUFFIXES:
        def f(window: str, seed: int) -> str:
            return f"submission_cb90_d6_{window}_seed{seed}_{suffix}.csv"

        average(
            [f("rows1500k", 2027), f("rows2m", 77)],
            f"submission_window_15m2027_2m77_{suffix}.csv",
        )
        average(
            [f("rows1500k", 2027), f("rows2m", 2027)],
            f"submission_window_15m2027_2m2027_{suffix}.csv",
        )
        average(
            [f("rows1500k", 2027), f("rows2m", 77), f("rows1m", 77)],
            f"submission_window_15m2027_2m77_1m77_{suffix}.csv",
            weights=[0.5, 0.25, 0.25],
        )
        average(
            [f("rows1500k", 2027), f("rows2m", 42), f("rows2m", 77), f("rows2m", 2027), f("rows2m", 9001)],
            f"submission_window_15m2027_2m_ens4_{suffix}.csv",
            weights=[0.5, 0.125, 0.125, 0.125, 0.125],
        )
        average(
            [f("rows2m", seed) for seed in (42, 77, 123, 2027, 9001)],
            f"submission_window_2m_ens5_{suffix}.csv",
        )
        made += 5
    print({"made": made})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

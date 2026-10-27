"""Build candidate-only blends around the corrected Lossguide63 winner."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
SUFFIXES = ("blend0.2", "blend0.3", "blend0.4", "blend0.5", "blend0.6", "raw")


def avg(paths: list[Path], output: Path, weights: list[float] | None = None) -> None:
    frames = [pd.read_csv(p) for p in paths]
    ids = frames[0].ID
    if any(not ids.equals(frame.ID) for frame in frames[1:]):
        raise ValueError(f"ID order mismatch: {output.name}")
    out = frames[0].copy()
    out.Target = np.average(np.stack([f.Target.to_numpy(float) for f in frames]), axis=0, weights=weights)
    out.to_csv(output, index=False)


def main() -> int:
    made = 0
    for suffix in SUFFIXES:
        winner = SUB / f"submission_cb90_all_lossguide63_seed2027_{suffix}.csv"
        stable = SUB / f"submission_cb90_all_lossguide63_seed77_{suffix}.csv"
        recency = SUB / f"submission_cb90_d6_rows1500k_seed2027_{suffix}.csv"
        window2m = SUB / f"submission_cb90_d6_rows2m_seed2027_{suffix}.csv"
        ens = [SUB / f"submission_cb90_all_lossguide63_seed{s}_{suffix}.csv" for s in (42, 77, 123, 2027, 9001)]
        avg([winner, stable], SUB / f"submission_lossguide63_2027_77_{suffix}.csv")
        avg([winner, recency], SUB / f"submission_lossguide63_2027_recency15m_{suffix}.csv", weights=[0.7, 0.3])
        avg([winner, recency], SUB / f"submission_lossguide63_2027_recency15m50_{suffix}.csv")
        avg([winner, window2m], SUB / f"submission_lossguide63_2027_recency2m_{suffix}.csv", weights=[0.7, 0.3])
        avg([winner, stable, recency, window2m], SUB / f"submission_lossguide63_window_ens_{suffix}.csv", weights=[0.5, 0.2, 0.2, 0.1])
        avg(ens, SUB / f"submission_lossguide63_ens5_{suffix}.csv")
        made += 6
    print({"made": made})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

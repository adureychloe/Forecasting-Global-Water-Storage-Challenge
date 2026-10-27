"""Average the completed five-head horizon predictions across seeds."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
PREFIX = "submission_horizon_variant_age0_1_2_3_6_7p"
SEEDS = (77, 2027, 123, 42, 9001)
SUFFIXES = ("blend0.3", "blend0.4", "blend0.6", "raw")


def main() -> int:
    made = 0
    for suffix in SUFFIXES:
        paths = [
            SUB / f"{PREFIX}_seed{s}_{suffix}.csv"
            if (SUB / f"{PREFIX}_seed{s}_{suffix}.csv").exists()
            else SUB / f"{PREFIX}_extra_seed{s}_{suffix}.csv"
            for s in SEEDS
        ]
        if not all(path.exists() for path in paths):
            raise FileNotFoundError([str(path) for path in paths if not path.exists()])
        frames = [pd.read_csv(path) for path in paths]
        ids = frames[0].iloc[:, 0]
        if any(not ids.equals(frame.iloc[:, 0]) for frame in frames[1:]):
            raise ValueError(f"ID order mismatch for {suffix}")
        out = frames[0].copy()
        out.iloc[:, 1] = np.mean([frame.iloc[:, 1].to_numpy(float) for frame in frames], axis=0)
        out.to_csv(SUB / f"submission_horizon5_ens5_{suffix}.csv", index=False)
        made += 1
    # A conservative cross-family candidate: 70% five-head ensemble plus 30%
    # of the already LB-validated four-head winner.
    five = pd.read_csv(SUB / "submission_horizon5_ens5_blend0.3.csv")
    four = pd.read_csv(SUB / "submission_horizon_heads_seed2027_blend0.3.csv")
    if not five.iloc[:, 0].equals(four.iloc[:, 0]):
        raise ValueError("ID order mismatch for cross-family blend")
    cross = five.copy()
    cross.iloc[:, 1] = 0.7 * five.iloc[:, 1].to_numpy(float) + 0.3 * four.iloc[:, 1].to_numpy(float)
    cross.to_csv(SUB / "submission_horizon5_ens5_plus_fourhead_blend0.7.csv", index=False)
    made += 1
    print({"made": made})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

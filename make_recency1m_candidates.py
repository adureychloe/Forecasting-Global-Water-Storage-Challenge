"""Build 1M-window ensemble and all-history cross-family candidates."""

from pathlib import Path

import numpy as np
import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"
PREFIX = "submission_horizon_recency1m_age0_1_2_3_6_7p"
SUFFIXES = ("blend0.3", "blend0.4", "blend0.6", "raw")


def main():
    made = 0
    for suffix in SUFFIXES:
        frames = [pd.read_csv(SUB / f"{PREFIX}_seed{s}_{suffix}.csv") for s in (2027, 77)]
        out = frames[0].copy()
        out.iloc[:, 1] = np.mean([f.iloc[:, 1].to_numpy(float) for f in frames], axis=0)
        out.to_csv(SUB / f"submission_horizon_recency1m_ens2_{suffix}.csv", index=False)
        allhist = pd.read_csv(SUB / f"submission_horizon5_ens5_{suffix}.csv")
        cross = out.copy()
        cross.iloc[:, 1] = 0.5 * out.iloc[:, 1].to_numpy(float) + 0.5 * allhist.iloc[:, 1].to_numpy(float)
        cross.to_csv(SUB / f"submission_horizon_recency1m_allhist50_{suffix}.csv", index=False)
        made += 2
    print({"made": made})


if __name__ == "__main__":
    main()

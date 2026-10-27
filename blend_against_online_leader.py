"""Blend a candidate family against the confirmed online leader."""

from pathlib import Path

import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def main():
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    families = {
        "lgbm_enhanced_seed2027": "submission_lgbm_horizon_enhanced_exact_seed2027_blend0.3.csv",
        "lgbm_enhanced_ens2": "submission_onlinecontrol_lgbm_enhanced_exact_ens2_w05.csv",
    }
    made = []
    for name, filename in families.items():
        pred = pd.read_csv(SUB / filename)
        if not leader.iloc[:, 0].equals(pred.iloc[:, 0]):
            raise ValueError(f"ID mismatch: {filename}")
        for w in (0.025, 0.05, 0.075, 0.10, 0.15):
            out = leader.copy()
            out.iloc[:, 1] = (1.0 - w) * leader.iloc[:, 1].to_numpy(float) + w * pred.iloc[:, 1].to_numpy(float)
            path = SUB / f"submission_onlineleader_{name}_w{int(w * 1000):03d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
    print({"made": made, "rows": len(leader)})


if __name__ == "__main__":
    main()

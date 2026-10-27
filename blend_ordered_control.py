"""Create conservative Ordered horizon blends against the online-best control."""

from pathlib import Path

import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def main():
    control = pd.read_csv(SUB / "submission_horizon_heads_seed2027_blend0.3.csv")
    p77 = pd.read_csv(SUB / "submission_horizon_ordered_l2_5_seed77_blend0.3.csv")
    p2027 = pd.read_csv(SUB / "submission_horizon_ordered_l2_5_seed2027_blend0.3.csv")
    ids = control.iloc[:, 0]
    if not p77.iloc[:, 0].equals(ids) or not p2027.iloc[:, 0].equals(ids):
        raise ValueError("ID order mismatch")
    preds = {
        "seed77": p77.iloc[:, 1].to_numpy(float),
        "ens2": 0.5 * (p77.iloc[:, 1].to_numpy(float) + p2027.iloc[:, 1].to_numpy(float)),
    }
    made = []
    for source, pred in preds.items():
        for w in (0.05, 0.10, 0.15, 0.20, 0.30):
            out = control.copy()
            out.iloc[:, 1] = (1 - w) * control.iloc[:, 1].to_numpy(float) + w * pred
            path = SUB / f"submission_onlinecontrol_ordered_l2_5_{source}_w{int(w * 100):02d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
    print({"made": made, "rows": len(control)})


if __name__ == "__main__":
    main()

"""Build conservative Plain/online-best control blends."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUB = ROOT / "submission"


def read(name):
    return pd.read_csv(SUB / name)


def main():
    control = read("submission_horizon_heads_seed2027_blend0.3.csv")
    p2027 = read("submission_horizon_plain_seed2027_blend0.3.csv")
    p77 = read("submission_horizon_plain_seed77_blend0.3.csv")
    ids = control.iloc[:, 0]
    for frame in (p2027, p77):
        if not frame.iloc[:, 0].equals(ids):
            raise ValueError("ID order mismatch")
    plain = 0.5 * p2027.iloc[:, 1].to_numpy(float) + 0.5 * p77.iloc[:, 1].to_numpy(float)
    made = []
    for name, pred in (("plain_seed77", p77.iloc[:, 1].to_numpy(float)), ("plain_ens2", plain)):
        for w in (0.05, 0.10, 0.15, 0.25):
            out = control.copy()
            out.iloc[:, 1] = (1 - w) * control.iloc[:, 1].to_numpy(float) + w * pred
            path = SUB / f"submission_onlinecontrol_plain_{name}_w{int(w * 100):02d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
    print({"made": made, "rows": len(control)})


if __name__ == "__main__":
    main()

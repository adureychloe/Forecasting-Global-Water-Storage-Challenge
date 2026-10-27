"""Conservative blends for the enhanced public-feature LightGBM horizon model."""

from pathlib import Path

import pandas as pd


SUB = Path(__file__).resolve().parent / "submission"


def main():
    control = pd.read_csv(SUB / "submission_horizon_heads_seed2027_blend0.3.csv")
    lgb = pd.read_csv(SUB / "submission_lgbm_horizon_enhanced_exact_seed2027_blend0.3.csv")
    lgb77 = pd.read_csv(SUB / "submission_lgbm_horizon_enhanced_exact_seed77_blend0.3.csv")
    if not control.iloc[:, 0].equals(lgb.iloc[:, 0]) or not control.iloc[:, 0].equals(lgb77.iloc[:, 0]):
        raise ValueError("ID order mismatch")
    p = lgb.iloc[:, 1].to_numpy(float)
    p77 = lgb77.iloc[:, 1].to_numpy(float)
    preds = {"seed2027": p, "ens2": 0.5 * (p + p77)}
    made = []
    for source, pred in preds.items():
        for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            out = control.copy()
            out.iloc[:, 1] = (1.0 - w) * control.iloc[:, 1].to_numpy(float) + w * pred
            path = SUB / f"submission_onlinecontrol_lgbm_enhanced_exact_{source}_w{int(w * 100):02d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
    print({"made": made, "rows": len(control)})


if __name__ == "__main__":
    main()

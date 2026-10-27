"""Create conservative online-leader blends for Bayesian CatBoost candidates."""

from pathlib import Path
import pandas as pd

SUB = Path(__file__).resolve().parent / "submission"


def main() -> None:
    leader = pd.read_csv(SUB / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    made = []
    families = {
        "bayes": lambda seed: SUB / f"submission_catboost_horizon_enhanced_ordered_bayes_seed{seed}_blend0.3.csv",
        "bayes_lowtemp": lambda seed: SUB / f"submission_catboost_horizon_enhanced_ordered_bayes_lowtemp_seed{seed}_blend0.3.csv",
        "plain_bayes": lambda seed: SUB / f"submission_catboost_horizon_enhanced_plain_bayes_seed{seed}_blend0.3.csv",
    }
    seed_groups = {"bayes": (321, 777), "bayes_lowtemp": (42, 9001), "plain_bayes": (123, 2027)}
    for family, seeds in seed_groups.items():
      for seed in seeds:
        source = families[family](seed)
        if not source.exists():
            continue
        pred = pd.read_csv(source)
        if not leader.iloc[:, 0].equals(pred.iloc[:, 0]):
            raise ValueError(source.name)
        for weight in (0.025, 0.05, 0.075):
            out = leader.copy()
            out.iloc[:, 1] = (1 - weight) * leader.iloc[:, 1].to_numpy(float) + weight * pred.iloc[:, 1].to_numpy(float)
            path = SUB / f"submission_onlineleader_catboost_{family}_seed{seed}_w{int(weight * 1000):03d}.csv"
            out.to_csv(path, index=False)
            made.append(path.name)
    print({"made": made, "rows": len(leader)})


if __name__ == "__main__":
    main()

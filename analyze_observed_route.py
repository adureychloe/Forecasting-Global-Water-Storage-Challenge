from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT / "data" / "Train.csv")
train["time"] = pd.to_datetime(train["time"])
mask_frame = pd.read_csv(ROOT / "data" / "Test.csv", usecols=["time", "TWS_t_masked"])
mask_frame["time"] = pd.to_datetime(mask_frame["time"])
rates = mask_frame.groupby("time")["TWS_t_masked"].mean().sort_index().to_numpy()
for fold in (0, 1):
    max_date = train["time"].max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = train.loc[(train["time"] >= cutoff) & (train["time"] <= max_date)].copy()
    rng = np.random.default_rng(42 + fold)
    date_rates = dict(zip(np.sort(valid["time"].unique()), rates[-valid["time"].nunique():]))
    row_rates = valid["time"].map(date_rates).fillna(float(np.mean(rates))).to_numpy()
    masked = np.zeros(len(valid), dtype=bool)
    for rate in np.unique(row_rates):
        idx = np.flatnonzero(row_rates == rate)
        masked[idx] = True if rate >= 0.9 else (False if rate <= 0.1 else rng.random(len(idx)) < rate)
    y = valid["target"].to_numpy(dtype=float)
    current = valid["TWS_t"].to_numpy(dtype=float)
    base = np.load(ROOT / "results" / f"fold{fold}" / "direct_base.npy")
    for weight in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        pred = base.copy()
        pred[~masked] = (1.0 - weight) * base[~masked] + weight * current[~masked]
        print(fold, weight, np.sqrt(np.mean((y - pred) ** 2)), np.sqrt(np.mean((y[~masked] - pred[~masked]) ** 2)))

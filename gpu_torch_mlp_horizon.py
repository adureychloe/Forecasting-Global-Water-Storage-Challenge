"""Lightweight PyTorch MLP horizon model for cross-family diversity."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import gpu_horizon_runner as horizon
import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
PREFIX = "torch_mlp_horizon"
LOG = RESULTS / f"{PREFIX}.jsonl"
SEEDS = (77, 2027)
BUCKETS = ("observed", "gap1", "gap2", "gap3_6", "gap7plus")


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def bucket(age):
    return np.select([age <= 0.5, age <= 1.5, age <= 2.5, age <= 6.5], BUCKETS[:4], default=BUCKETS[4])


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def score(y, p, mask):
    return {"all": rmse(y, p), "masked": rmse(y[mask], p[mask]), "observed": rmse(y[~mask], p[~mask])}


class Net(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_input(frame, b):
    x = frame.to_numpy(dtype=np.float32, copy=True)
    one = np.zeros((len(x), len(BUCKETS)), dtype=np.float32)
    for i, name in enumerate(BUCKETS):
        one[:, i] = b == name
    x = np.concatenate([x, one], axis=1)
    x[~np.isfinite(x)] = 0.0
    mu = np.nanmedian(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd < 1e-5] = 1.0
    x = (x - mu) / sd
    return np.clip(x, -8.0, 8.0).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def fit_predict(xfit, yfit, xval, bfit, bval, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xf, mu, sd = make_input(xfit, bfit)
    xv = xval.to_numpy(dtype=np.float32, copy=True)
    xv = np.concatenate([xv, np.stack([(bval == name) for name in BUCKETS], axis=1).astype(np.float32)], axis=1)
    xv[~np.isfinite(xv)] = 0.0
    xv = np.clip((xv - mu) / sd, -8.0, 8.0).astype(np.float32)
    model = Net(xf.shape[1]).to(device)
    ds = TensorDataset(torch.from_numpy(xf), torch.from_numpy(yfit.astype(np.float32)))
    loader = DataLoader(ds, batch_size=8192, shuffle=True, num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    model.train()
    for epoch in range(6):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach())
        log({"stage": "epoch", "seed": seed, "epoch": epoch + 1, "loss": total / max(1, len(loader))})
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xv).to(device)).cpu().numpy().astype(np.float32)
    del model, loader, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def main():
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time); test.time = pd.to_datetime(test.time)
    states = public_states(test)
    prepared = {}
    log({"stage": "start", "states": states.astype(int).tolist(), "buckets": BUCKETS})
    for seed in SEEDS:
        folds = []
        for fold in (0, 1):
            if fold not in prepared:
                prepared[fold] = horizon.fold_data(train, states, fold)
            xfit, yfit, xval, yval, bfit, bval, cutoff, max_date = prepared[fold]
            pred = fit_predict(xfit, yfit, xval, bfit, bval, seed)
            mask = bval != "observed"
            base = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
            blends = {str(w): score(yval, w * pred + (1 - w) * base, mask) for w in np.arange(0.1, 1.01, 0.1)}
            best = min(blends, key=lambda w: blends[w]["all"])
            row = {"stage": "fold", "seed": seed, "fold": fold, "raw": score(yval, pred, mask), "best_weight": float(best), "best_blend": blends[best], "period": [str(cutoff.date()), str(max_date.date())]}
            log(row); folds.append(row)
        mean = float(np.mean([r["best_blend"]["all"] for r in folds]))
        log({"stage": "pair", "seed": seed, "mean": mean})
        train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
        test_input = test.copy(); test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
        combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
        features = core.build_features(combined).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        xtrain, xtest = features.iloc[:len(train)], features.iloc[len(train):]
        btrain, btest = bucket(xtrain["TWS_t_age"].to_numpy()), bucket(xtest["TWS_t_age"].to_numpy())
        pred = fit_predict(xtrain, train.target.to_numpy(np.float32), xtest, btrain, btest, seed)
        base = pd.read_csv(SUBMISSIONS / "submission_aug35_ageflag_ens3_ridge30_a100.csv").iloc[:, 1].to_numpy(float)
        for w in (0.3, 0.4, 0.6, 1.0):
            vals = pred if w == 1 else w * pred + (1 - w) * base
            suffix = "raw" if w == 1 else f"blend{w:.1f}"
            core.make_submission_frame(test, vals, sample).to_csv(SUBMISSIONS / f"submission_{PREFIX}_seed{seed}_{suffix}.csv", index=False)
        log({"stage": "full", "seed": seed, "mean": mean})
    log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

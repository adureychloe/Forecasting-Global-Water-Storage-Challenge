"""Twelve-month temporal CNN for leakage-safe panel forecasting diversity."""

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

import gpu_nightly_runner as core
from gpu_dualhead_runner import mask_by_template, public_states


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUBMISSIONS = ROOT / "submission"
LOG = RESULTS / "torch_tcn_sequence.jsonl"
PREFIX = "torch_tcn_sequence"
SEQ_COLS = ("TWS_t", "SPEI_01_t", "SPEI_03_t", "SPEI_06_t", "SPEI_12_t", "SOIL_MOISTURE_t")
SEQ_LEN = 12
SEED = 2027


def log(row):
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print(json.dumps(payload), flush=True)


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def score(y, pred, masked):
    return {
        "all": rmse(y, pred),
        "masked": rmse(y[masked], pred[masked]),
        "observed": rmse(y[~masked], pred[~masked]),
    }


def make_arrays(frame: pd.DataFrame, local_normalize: bool = False):
    work = frame.reset_index(drop=True).copy()
    work["time"] = pd.to_datetime(work["time"])
    work["_position"] = np.arange(len(work))
    ordered = work.sort_values(["lat", "lon", "time"], kind="mergesort")
    grouped = ordered.groupby(["lat", "lon"], sort=False, observed=True)
    channels = []
    for column in SEQ_COLS:
        history = []
        for lag in range(SEQ_LEN - 1, -1, -1):
            values = ordered[column] if lag == 0 else grouped[column].shift(lag)
            history.append(values.to_numpy(dtype=np.float32))
        channels.append(np.stack(history, axis=1))
    tws_missing = ~np.isfinite(channels[0])
    local_static = None
    if local_normalize:
        visible = ~tws_missing
        count = visible.sum(axis=1)
        tws = np.where(visible, channels[0], 0.0)
        center = np.divide(
            tws.sum(axis=1),
            count,
            out=np.zeros(len(tws), dtype=np.float32),
            where=count > 0,
        )
        variance = np.divide(
            ((tws - center[:, None]) ** 2 * visible).sum(axis=1),
            count,
            out=np.ones(len(tws), dtype=np.float32),
            where=count > 1,
        )
        scale = np.sqrt(np.maximum(variance, 0.01)).astype(np.float32)
        channels[0] = np.where(
            visible,
            (tws - center[:, None]) / scale[:, None],
            0.0,
        ).astype(np.float32)
        local_static = np.column_stack((center, np.log1p(scale))).astype(np.float32)
    channels.append(tws_missing.astype(np.float32))
    sequence = np.stack(channels, axis=1)
    sequence[~np.isfinite(sequence)] = 0.0
    lat = np.deg2rad(ordered["lat"].to_numpy(dtype=np.float32))
    lon = np.deg2rad(ordered["lon"].to_numpy(dtype=np.float32))
    month = ordered["time"].dt.month.to_numpy(dtype=np.float32)
    age = np.cumprod(tws_missing[:, ::-1], axis=1).sum(axis=1).astype(np.float32)
    static = np.column_stack(
        (
            np.sin(lat),
            np.cos(lat),
            np.sin(lon),
            np.cos(lon),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            age,
        )
    ).astype(np.float32)
    if local_static is not None:
        static = np.column_stack((static, local_static)).astype(np.float32)
    order = np.argsort(ordered["_position"].to_numpy())
    return sequence[order], static[order]


class TCN(nn.Module):
    def __init__(self, channels: int, static_features: int):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + static_features, 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 1),
        )

    def forward(self, sequence, static):
        encoded = self.temporal(sequence)[:, :, -1]
        return self.head(torch.cat((encoded, static), dim=1)).squeeze(1)


def fit_predict(x_fit, static_fit, y_fit, x_query, static_query, seed=SEED, epochs=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    channel_mu = x_fit.mean(axis=(0, 2), keepdims=True)
    channel_sd = x_fit.std(axis=(0, 2), keepdims=True)
    channel_sd[channel_sd < 1e-5] = 1.0
    static_mu = static_fit.mean(axis=0, keepdims=True)
    static_sd = static_fit.std(axis=0, keepdims=True)
    static_sd[static_sd < 1e-5] = 1.0
    xf = np.clip((x_fit - channel_mu) / channel_sd, -8, 8).astype(np.float32)
    xq = np.clip((x_query - channel_mu) / channel_sd, -8, 8).astype(np.float32)
    sf = np.clip((static_fit - static_mu) / static_sd, -8, 8).astype(np.float32)
    sq = np.clip((static_query - static_mu) / static_sd, -8, 8).astype(np.float32)
    y_mu = float(np.mean(y_fit))
    y_sd = max(float(np.std(y_fit)), 1e-5)
    yn = ((y_fit - y_mu) / y_sd).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(xf), torch.from_numpy(sf), torch.from_numpy(yn))
    loader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=0, pin_memory=True)
    model = TCN(xf.shape[1], sf.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for sequence, static, target in loader:
            sequence = sequence.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(sequence, static), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
        log({"stage": "epoch", "epoch": epoch + 1, "loss": total / len(loader)})
    model.eval()
    predictions = []
    query = TensorDataset(torch.from_numpy(xq), torch.from_numpy(sq))
    query_loader = DataLoader(query, batch_size=16384, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for sequence, static in query_loader:
            pred = model(sequence.to(device, non_blocking=True), static.to(device, non_blocking=True))
            predictions.append(pred.cpu().numpy())
    result = np.concatenate(predictions).astype(np.float32) * y_sd + y_mu
    del model, loader, dataset, query_loader, query, xf, xq, sf, sq
    gc.collect()
    torch.cuda.empty_cache()
    return result


def fold_data(train, states, fold):
    max_date = train.time.max() - pd.DateOffset(months=12 * fold)
    cutoff = max_date - pd.DateOffset(months=12) + pd.DateOffset(days=1)
    valid = train.loc[(train.time >= cutoff) & (train.time <= max_date)].copy()
    fit = train.loc[train.time < cutoff].copy()
    combined = pd.concat([fit.drop(columns=["target"]), valid.drop(columns=["target"])], ignore_index=True, sort=False)
    combined = mask_by_template(combined, states, align_validation=True)
    sequence, static = make_arrays(combined)
    n = len(fit)
    valid_masked = combined.TWS_t_masked.iloc[n:].to_numpy(dtype=bool)
    return sequence[:n], static[:n], fit.target.to_numpy(np.float32), sequence[n:], static[n:], valid.target.to_numpy(np.float32), valid_masked, cutoff, max_date


def main():
    train = pd.read_csv(core.DATA / "Train.csv")
    test = pd.read_csv(core.DATA / "Test.csv")
    sample = pd.read_csv(core.DATA / "SampleSubmission.csv")
    train.time = pd.to_datetime(train.time)
    test.time = pd.to_datetime(test.time)
    states = public_states(test)
    rows = []
    log({"stage": "start", "sequence_length": SEQ_LEN, "channels": list(SEQ_COLS)})
    for fold in (0, 1):
        xfit, sfit, yfit, xvalid, svalid, yvalid, masked, cutoff, max_date = fold_data(train, states, fold)
        pred = fit_predict(xfit, sfit, yfit, xvalid, svalid)
        baseline = np.load(RESULTS / f"fold{fold}" / "direct_base.npy")
        blends = {str(w): score(yvalid, w * pred + (1 - w) * baseline, masked) for w in np.arange(0.05, 0.51, 0.05)}
        best = min(blends, key=lambda weight: blends[weight]["all"])
        row = {"stage": "fold", "fold": fold, "raw": score(yvalid, pred, masked), "best_weight": float(best), "best_blend": blends[best], "period": [str(cutoff.date()), str(max_date.date())]}
        log(row)
        rows.append(row)
        del xfit, sfit, yfit, xvalid, svalid, yvalid
        gc.collect()
    mean = float(np.mean([row["best_blend"]["all"] for row in rows]))
    log({"stage": "pair", "seed": SEED, "mean": mean})
    train_input = mask_by_template(train.drop(columns=["target"]), states, align_validation=False)
    test_input = test.copy()
    test_input.loc[test_input.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
    combined = pd.concat([train_input, test_input], ignore_index=True, sort=False)
    sequence, static = make_arrays(combined)
    n = len(train)
    pred = fit_predict(sequence[:n], static[:n], train.target.to_numpy(np.float32), sequence[n:], static[n:])
    leader = pd.read_csv(SUBMISSIONS / "submission_onlinecontrol_ordered_l2_5_seed77_w05.csv")
    for weight in (0.05, 0.10, 0.20, 1.0):
        values = pred if weight == 1 else weight * pred + (1 - weight) * leader.iloc[:, 1].to_numpy(float)
        suffix = "raw" if weight == 1 else f"onlineleader_w{int(weight * 100):02d}"
        core.make_submission_frame(test, values, sample).to_csv(SUBMISSIONS / f"submission_{PREFIX}_{suffix}.csv", index=False)
    log({"stage": "full", "seed": SEED, "mean": mean})
    log({"stage": "complete"})
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()

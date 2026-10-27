"""CPU physical baselines on the official sparse calendar.

Compares last-obs persist, recursive persist, cell-month climatology, and
persist+ERA5 water-balance increment against #262 aligned 0.522 / static 0.526.
lat/lon are lookup keys only. No GRACE. No submission.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from era5_cell_features import load_or_build_table
from sparse_replay import calendar_template, make_replay, score_predictions

ANCHORS = ("2007-09", "2008-06", "2009-01")


def last_obs_map(history: pd.DataFrame) -> pd.DataFrame:
    hist = history.dropna(subset=["TWS_t"]).sort_values("time")
    last = hist.groupby(["lat", "lon"], sort=False).tail(1)[["lat", "lon", "TWS_t"]]
    return last.rename(columns={"TWS_t": "persist"})


def climatology_map(fit: pd.DataFrame) -> pd.DataFrame:
    work = fit.dropna(subset=["TWS_t"]).copy()
    work["month"] = pd.to_datetime(work.time).dt.month
    clim = work.groupby(["lat", "lon", "month"], sort=False).TWS_t.mean().reset_index()
    return clim.rename(columns={"TWS_t": "clim"})


def merge_wb(query: pd.DataFrame, era5: pd.DataFrame) -> np.ndarray:
    q = query.copy()
    q["_month_key"] = q.time.dt.year.to_numpy(np.int32) * 12 + q.time.dt.month.to_numpy(np.int32)
    cols = ["lat", "lon", "_month_key", "wb"]
    mapped = q[["lat", "lon", "_month_key"]].merge(era5[cols], on=["lat", "lon", "_month_key"], how="left")
    wb = mapped["wb"].to_numpy(np.float64)
    return np.where(np.isfinite(wb), wb, 0.0)


def predict_persist(history, current):
    last = last_obs_map(history)
    mapped = current[["lat", "lon"]].merge(last, on=["lat", "lon"], how="left")
    values = mapped["persist"].to_numpy(np.float64)
    fallback = pd.to_numeric(current.get("TWS_t"), errors="coerce").to_numpy(np.float64)
    return np.where(np.isfinite(values), values, np.where(np.isfinite(fallback), fallback, 0.0)).astype(np.float32)


def walk_persist(fit, query, recursive: bool):
    from aligned_feedback import walk_aligned

    train_input = fit.drop(columns=["target"], errors="ignore")
    if recursive:
        return walk_aligned(train_input, query, predict_persist)
    pred = np.empty(len(query), dtype=np.float32)
    for month, group in query.groupby("time", sort=True):
        pred[group.index.to_numpy()] = predict_persist(train_input, group)
    return pred


def main() -> None:
    train = pd.read_csv("data/Train.csv", parse_dates=["time"])
    template = pd.read_csv("data/Test.csv", parse_dates=["time"])
    weights = calendar_template(template)["_offset"].value_counts().to_dict()
    era5 = load_or_build_table(pd.concat([train, template], ignore_index=True))
    rows = []
    for anchor in ANCHORS:
        fit, query, truth = make_replay(train, template, anchor)
        persist = walk_persist(fit, query, recursive=False)
        rec_persist = walk_persist(fit, query, recursive=True)
        clim = climatology_map(fit)
        q = query.copy()
        q["month"] = q.time.dt.month
        clim_pred = q.merge(clim, on=["lat", "lon", "month"], how="left")["clim"].to_numpy(np.float64)
        clim_pred = np.where(np.isfinite(clim_pred), clim_pred, persist).astype(np.float32)
        wb = merge_wb(query, era5)
        delta = truth.astype(np.float64) - persist.astype(np.float64)
        finite = np.isfinite(delta) & np.isfinite(wb) & (np.abs(wb) > 1e-12)
        if finite.sum() >= 1000:
            k = float(np.dot(wb[finite], delta[finite]) / np.dot(wb[finite], wb[finite]))
        else:
            k = 0.0
        wb_pred = (persist.astype(np.float64) + k * wb).astype(np.float32)
        for name, pred in (
            ("persist", persist),
            ("rec_persist", rec_persist),
            ("climatology", clim_pred),
            ("persist_wb", wb_pred),
        ):
            scores = score_predictions(query, truth, pred, weights)
            row = {"anchor": anchor, "name": name, "k": k, **scores}
            rows.append(row)
            print(row, flush=True)
    frame = pd.DataFrame(rows)

    def pooled(name, col):
        vals = frame.loc[frame.name == name, col].to_numpy(dtype=float)
        return float(np.mean(vals ** 2) ** 0.5)

    summary = {
        name: {
            "all": pooled(name, "all"),
            "masked": pooled(name, "masked"),
        }
        for name in ("persist", "rec_persist", "climatology", "persist_wb")
    }
    print({"pooled": summary, "ref_aligned": 0.522228, "ref_static": 0.526434}, flush=True)


if __name__ == "__main__":
    main()

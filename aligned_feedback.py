"""Time-aligned recursive feedback: write forecast(t+1) into month t+1, not t.

Legacy submissions store the t+1 forecast in the current month's TWS_t. That
makes lag-1 at the next consecutive test month equal to the current-month TWS
estimate, which is not how training features are built. This walker keeps the
current month's TWS_t as observed/persist/previous-aligned fill, and applies
the forecast only when the next query month is exactly one calendar month later.
Intervening months are not synthesized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def next_month_start(ts) -> pd.Timestamp:
    return (pd.Timestamp(ts).to_period("M") + 1).to_timestamp()


def walk_aligned(train_input, query, predict_month, on_month=None):
    if "target" in train_input or "target" in query:
        raise ValueError("target labels must not enter inference history")
    work = query.copy().reset_index(drop=True)
    work.loc[work.TWS_t_masked.astype(bool), "TWS_t"] = np.nan
    history = train_input.copy().reset_index(drop=True)
    predictions = np.full(len(work), np.nan, dtype=np.float32)
    pending = None
    for month, group in work.groupby("time", sort=True):
        month = pd.Timestamp(month)
        if not (history.time < month).all():
            raise ValueError("inference history contains current/future rows")
        current = group.copy()
        if pending is not None:
            pend_time, pend = pending
            if pend_time == month:
                mapped = current[["lat", "lon"]].merge(pend, on=["lat", "lon"], how="left")
                fill = mapped["_aligned"].to_numpy(dtype=np.float64)
                mask = current["TWS_t_masked"].to_numpy(dtype=bool)
                tws = pd.to_numeric(current["TWS_t"], errors="coerce").to_numpy(dtype=np.float64)
                current["TWS_t"] = np.where(mask & np.isfinite(fill), fill, tws)
            pending = None
        values = np.asarray(predict_month(history, current), dtype=np.float32)
        if values.shape != (len(current),) or not np.isfinite(values).all():
            raise ValueError("invalid monthly predictions")
        predictions[current.index.to_numpy()] = values
        history = pd.concat([history, current], ignore_index=True, sort=False)
        pending = (
            next_month_start(month),
            pd.DataFrame(
                {
                    "lat": current["lat"].to_numpy(),
                    "lon": current["lon"].to_numpy(),
                    "_aligned": values,
                }
            ),
        )
        if on_month is not None:
            on_month(current, values)
    return predictions

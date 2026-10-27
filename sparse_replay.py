"""Sparse-calendar evaluation primitives; no GPU imports or training side effects."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_template(template):
    out = template[['time', 'lat', 'lon', 'TWS_t_masked']].copy()
    periods = pd.to_datetime(out.time).dt.to_period('M')
    out['_offset'] = periods.astype('int64') - periods.min().ordinal
    if out.duplicated(['_offset', 'lat', 'lon']).any():
        raise ValueError('duplicate template cell/month')
    return out.drop(columns='time')


def make_replay(train, template, anchor):
    """Map exact template dates/cell masks onto history; exclude all intervening rows.

    Missing cells are intersected and reported by the caller. Missing whole months
    are rejected: neither time compression nor unseen TWS imputation is allowed.
    Validation labels are returned separately and never retained in query inputs.
    """
    start = pd.Period(anchor, freq='M')
    layout = calendar_template(template)
    layout['time'] = [ (start + int(d)).to_timestamp() for d in layout['_offset'] ]
    source = train.copy()
    source['time'] = pd.to_datetime(source.time)
    if source.duplicated(['time', 'lat', 'lon']).any():
        raise ValueError('duplicate training cell/month')
    fit = source.loc[source.time < start.to_timestamp()].reset_index(drop=True)
    query = layout.merge(source.drop(columns='TWS_t_masked', errors='ignore'),
                         on=['time', 'lat', 'lon'], how='inner', validate='one_to_one')
    if set(query['_offset']) != set(layout['_offset']):
        raise ValueError('historical replay is missing a requested month')
    query = query.sort_values(['time', 'lat', 'lon'], kind='stable').reset_index(drop=True)
    truth = query.pop('target').to_numpy(dtype=np.float32)
    query['TWS_t_masked'] = query.TWS_t_masked.astype(bool)
    query.loc[query.TWS_t_masked, 'TWS_t'] = np.nan
    return fit, query, truth


def walk_legacy(train_input, query, predict_month, on_month=None):
    """Replay the existing production feedback convention, intentionally unchanged.

    Existing submissions write forecast(t+1) into the masked history row at t.
    This reproduces that heuristic for a baseline, not a physically aligned state
    estimator. Corrected target-time feedback must be evaluated as a separate mode.
    The callback sees only past inputs and this month's query, never query labels.
    """
    if 'target' in train_input or 'target' in query:
        raise ValueError('target labels must not enter inference history')
    work = query.copy().reset_index(drop=True)
    work.loc[work.TWS_t_masked.astype(bool), 'TWS_t'] = np.nan
    history = train_input.copy().reset_index(drop=True)
    predictions = np.full(len(work), np.nan, dtype=np.float32)
    for month, group in work.groupby('time', sort=True):
        if not (history.time < month).all():
            raise ValueError('inference history contains current/future rows')
        current = group.copy()
        values = np.asarray(predict_month(history, current), dtype=np.float32)
        if values.shape != (len(current),) or not np.isfinite(values).all():
            raise ValueError('invalid monthly predictions')
        predictions[current.index] = values
        filled = current.copy()
        mask = filled.TWS_t_masked.astype(bool)
        filled.loc[mask, 'TWS_t'] = values[mask]
        history = pd.concat([history, filled], ignore_index=True, sort=False)
        if on_month is not None:
            on_month(current, values)
    return predictions


def score_predictions(query, truth, predictions, month_weights):
    y, p = np.asarray(truth, dtype=float), np.asarray(predictions, dtype=float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('invalid labels or predictions')
    counts = query['_offset'].value_counts()
    weight = query['_offset'].map(month_weights).to_numpy(float) / query['_offset'].map(counts).to_numpy(float)
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError('invalid month weights')
    error = (y - p) ** 2
    mask = query.TWS_t_masked.to_numpy(bool)

    def metric(select):
        return float(np.sqrt(np.average(error[select], weights=weight[select]))) if np.any(select) else None

    return {'all': metric(np.ones(len(query), bool)), 'masked': metric(mask),
            'observed': metric(~mask), 'rows': len(query)}


def promotion_gate(candidate, reference, min_gain=0.0005, max_fold_regression=0.002):
    """Fail closed unless complete, comparable paired folds improve reliably."""
    def reject(reason):
        return {'passed': False, 'reason': reason}

    if not reference:
        return reject('no reference: validation only')
    if not candidate.get('complete') or not reference.get('complete'):
        return reject('incomplete validation')
    if candidate.get('protocol') != reference.get('protocol'):
        return reject('protocol mismatch')
    current_rows, ref_rows = candidate['folds'], reference['folds']
    current = {r['key']: r for r in current_rows}
    base = {r['key']: r for r in ref_rows}
    protocol = candidate['protocol']
    expected = {f'{anchor}/{seed}' for anchor in protocol.get('anchors', [])
                for seed in protocol.get('seeds', [])}
    if not expected or current.keys() != expected or base.keys() != expected:
        return reject('missing planned folds')
    if (not current or current.keys() != base.keys() or len(current) != len(current_rows)
            or len(base) != len(ref_rows)):
        return reject('missing or duplicate paired folds')
    keys = sorted(current)
    c = np.array([[current[k][m] for m in ('all', 'masked')] for k in keys], dtype=float)
    b = np.array([[base[k][m] for m in ('all', 'masked')] for k in keys], dtype=float)
    if not np.isfinite(c).all() or not np.isfinite(b).all():
        return reject('nonfinite scores')
    if ((c - b) > max_fold_regression).any():
        return reject('a paired fold or masked segment regressed')
    gains = np.sqrt(np.mean(b*b, axis=0)) - np.sqrt(np.mean(c*c, axis=0))
    if (gains < min_gain).any():
        return reject('insufficient overall or masked improvement')
    return {'passed': True, 'reason': 'paired validation passed',
            'overall_gain': float(gains[0]), 'masked_gain': float(gains[1])}

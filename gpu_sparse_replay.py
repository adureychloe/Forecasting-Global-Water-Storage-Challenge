"""Train/evaluate fixed CatBoost heads on the actual sparse Test calendar.

Default is CV-only. --full needs a complete comparable reference and a passing
paired gate. Legacy recursion deliberately reproduces the existing submissions;
the time-shifted feedback heuristic is not relabeled as a correct TWS state.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sparse_replay import calendar_template, make_replay, promotion_gate, score_predictions, walk_legacy

ROOT = Path(__file__).resolve().parent
INPUT_COLS = ['time', 'lat', 'lon', 'TWS_t', 'SPEI_01_t', 'SPEI_03_t',
              'SPEI_06_t', 'SPEI_12_t', 'SOIL_MOISTURE_t', 'month_sin', 'month_cos', 'TWS_t_masked']
BUCKETS = ('observed', 'gap1', 'gap2', 'gap3_6', 'gap7plus')


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inputs(frame):
    return frame[[c for c in INPUT_COLS if c in frame]].copy().reset_index(drop=True)


def clean(features):
    from coordinate_compliance import drop_coordinate_features
    features = features.drop(columns=[c for c in features if str(c).startswith('TWS_t_masked')], errors='ignore')
    return drop_coordinate_features(features).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def bucket(features):
    age = features['TWS_t_age'].to_numpy()
    return np.select([age <= .5, age <= 1.5, age <= 2.5, age <= 6.5], BUCKETS[:4], default=BUCKETS[4])


def transform(values, features, variant, inverse=False):
    if variant == 'raw':
        return np.asarray(values, dtype=np.float32)
    if variant == 'centered':
        import block_centered_target as module
    else:
        import block_robust_normalized_target as module
    return (module.inverse_predictions if inverse else module.transform_targets)(values, features)


def fit_heads(x_fit, y_fit, fit, start, seed, args, folder, log):
    from catboost import CatBoostRegressor
    import gpu_catboost_idw_era5_pattern as base
    selected = fit.time >= (pd.Timestamp(start) - pd.DateOffset(months=args.recency_months))
    groups = bucket(x_fit)
    models = {}
    for name in BUCKETS:
        idx = selected.to_numpy() & (groups == name)
        if idx.sum() < args.min_head_rows:
            log(stage='head_fallback', bucket=name, n_fit=int(idx.sum()))
            continue
        path = folder / f'{name}.cbm'
        if path.exists():
            model = CatBoostRegressor().load_model(str(path))
            log(stage='head_resumed', bucket=name)
        else:
            log(stage='head_start', bucket=name, n_fit=int(idx.sum()))
            model = base.model(seed, args.iterations)
            # Fixed full-fit budget: no outer validation labels used for early stopping.
            model.fit(x_fit.loc[idx], transform(y_fit[idx], x_fit.loc[idx], args.variant), verbose=False)
            model.save_model(str(path))
            log(stage='head_complete', bucket=name, iterations=args.iterations)
        models[name] = model
    return models


def predict_heads(models, features, variant):
    groups = bucket(features)
    fallback = features['tws_neighbor_anchor_mean'].to_numpy(dtype=np.float32).copy()
    for name, model in models.items():
        idx = groups == name
        if idx.any():
            fallback[idx] = transform(model.predict(features.loc[idx]), features.loc[idx], variant, inverse=True)
    if not np.isfinite(fallback).all():
        raise ValueError('nonfinite head prediction')
    return fallback


def prepare(fit, query, states, log):
    import gpu_catboost_idw_era5_pattern as base
    from gpu_dualhead_runner import mask_by_template
    train_input = mask_by_template(inputs(fit), states, align_validation=False)
    query_input = inputs(query)
    log(stage='features_start', fit_rows=len(fit), query_rows=len(query))
    combined = pd.concat([train_input, query_input], ignore_index=True, sort=False)
    features = clean(base.build_features(combined))
    n = len(fit)
    x_fit, x_query = base.augment_features(features.iloc[:n].copy(), fit.target.to_numpy(np.float32),
                                          features.iloc[n:].copy(), fit, query_input)
    x_fit, x_query = clean(x_fit).reset_index(drop=True), clean(x_query).reset_index(drop=True)
    log(stage='features_complete', feature_count=len(x_fit.columns))
    return train_input, query_input, x_fit, x_query


def rollout(models, train_input, query_input, static_features, args, folder, log, truth=None, offsets=None):
    from sparse_model_features import dynamic_query_features
    import neighborhood_tws_features as nb
    # Supervised features and all ERA5 values depend on frozen fit data and date,
    # not on recursive TWS. Reuse them exactly from the prepared query table.
    cached_columns = [c for c in static_features if c.startswith(('era5_', 'target_'))]
    static_pred = predict_heads(models, static_features, args.variant)
    np.save(folder / 'static.npy', static_pred)

    def predict_month(history, current):
        month_key = pd.Timestamp(current.time.iloc[0]).strftime('%Y-%m')
        saved = folder / f'month_{month_key}.npy'
        if saved.exists():
            return np.load(saved)
        features = clean(dynamic_query_features(history, current))
        features = pd.concat([
            features.drop(columns=cached_columns, errors='ignore'),
            static_features.loc[current.index, cached_columns].reset_index(drop=True),
        ], axis=1)
        features = features.reindex(columns=static_features.columns)
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError('query feature schema mismatch')
        if current.time.iloc[0] == query_input.time.min():
            expected = static_features.loc[current.index].to_numpy()
            if not np.allclose(features.to_numpy(), expected, rtol=1e-6, atol=1e-6):
                raise ValueError('first-month cached query differs from full-history features')
            log(stage='query_equivalence_passed', rows=len(current))
        routed = bucket(features)
        pred = predict_heads(models, features, args.variant)
        for _ in range(1, args.inner_iters):
            features = clean(nb.overlay_pred_neighborhood(current.reset_index(drop=True), features, pred))
            # Overlay must leave the router unchanged, as in the original model.
            if not np.array_equal(bucket(features), routed):
                raise ValueError('spatial overlay changed age routing')
            pred = predict_heads(models, features, args.variant)
        np.save(saved, pred)
        return pred

    def on_month(current, pred):
        entry = {'stage': 'month_complete', 'month': str(current.time.iloc[0].date()), 'rows': len(current)}
        if truth is not None:
            idx = current.index.to_numpy()
            error = pred.astype(float) - truth[idx]
            mask = current.TWS_t_masked.to_numpy(bool)
            entry.update(rmse=float(np.sqrt(np.mean(error ** 2))),
                         masked_rmse=float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None,
                         offset=int(offsets[idx[0]]))
        log(**entry)
        gc.collect()

    predictions = walk_legacy(train_input, query_input, predict_month, on_month)
    np.save(folder / 'recursive.npy', predictions)
    return static_pred, predictions


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--anchors', default='2007-09,2008-06,2009-01')
    p.add_argument('--seeds', default='2027,77')
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--recency-months', type=int, default=68)
    p.add_argument('--inner-iters', type=int, default=3)
    p.add_argument('--min-head-rows', type=int, default=1000)
    p.add_argument('--variant', choices=('raw', 'centered', 'robustnorm'), default='raw')
    p.add_argument('--reference', type=Path)
    p.add_argument('--full', action='store_true')
    p.add_argument('--max-cells', type=int, default=0, help='smoke test only; prohibits --full')
    p.add_argument('--preflight', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.full and (not args.reference or args.max_cells):
        raise SystemExit('--full requires a reference and all cells')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        raise SystemExit('run-id must be a directory name')
    if min(args.iterations, args.recency_months, args.inner_iters, args.min_head_rows) <= 0:
        raise SystemExit('training limits must be positive')
    folder = ROOT / 'results' / 'sparse_replay' / args.run_id
    folder.mkdir(parents=True, exist_ok=True)

    def log(**row):
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), **row}
        text = json.dumps(row, allow_nan=False)
        with (folder / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(text + '\n')
        print(text, flush=True)

    train = pd.read_csv(ROOT / 'data' / 'Train.csv', parse_dates=['time'])
    template = pd.read_csv(ROOT / 'data' / 'Test.csv', parse_dates=['time'])
    official_template = template.copy()
    layout = calendar_template(official_template)
    states = official_template.groupby('time', sort=True).TWS_t_masked.mean().ge(.5).to_numpy()
    if args.max_cells:
        # A deterministic geographically compact panel keeps neighbors meaningful.
        cells = train[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).head(args.max_cells)
        train = train.merge(cells, on=['lat', 'lon'])
        template = template.merge(cells, on=['lat', 'lon'])
    anchors, seeds = args.anchors.split(','), [int(s) for s in args.seeds.split(',')]
    if len(set(anchors)) != len(anchors) or len(set(seeds)) != len(seeds):
        raise SystemExit('duplicate anchors or seeds')
    protocol = {'version': 'sparse-calendar-legacy-recursion-v1', 'anchors': anchors, 'seeds': seeds,
                'offsets': sorted(int(i) for i in layout['_offset'].unique()),
                'iterations': args.iterations, 'recency_months': args.recency_months,
                'inner_iters': args.inner_iters, 'max_cells': args.max_cells, 'min_head_rows': args.min_head_rows,
                'train_sha256': digest(ROOT / 'data' / 'Train.csv'),
                'test_sha256': digest(ROOT / 'data' / 'Test.csv'),
                'metric': 'test-month-weighted-MSE; equal-fold-pooled-MSE'}
    modules = ['gpu_sparse_replay.py', 'sparse_replay.py', 'sparse_model_features.py',
               'gpu_catboost_idw_era5_pattern.py', 'baseline.py', 'enhanced_features.py',
               'neighborhood_tws_features.py', 'era5_cell_features.py', 'era5_anomaly_features.py',
               'target_stats_features.py', 'target_stats_blockfrozen_features.py',
               'block_centered_target.py', 'block_robust_normalized_target.py', 'coordinate_compliance.py',
               'gpu_dualhead_runner.py', 'gpu_horizon_runner.py', 'gpu_nightly_runner.py']
    manifest = {'protocol': protocol, 'variant': args.variant,
                'code_sha256': {name: digest(ROOT / name) for name in modules}}
    manifest_path = folder / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise SystemExit('run-id already contains a different recipe; choose a new run-id')
    atomic_json(manifest_path, manifest)
    month_weights = layout['_offset'].value_counts().to_dict()
    for anchor in anchors:
        fit, query, _ = make_replay(train, template, anchor)
        log(stage='preflight', anchor=anchor, fit_rows=len(fit), query_rows=len(query),
            coverage=len(query)/len(template), months=query.time.dt.strftime('%Y-%m').unique().tolist(),
            mask_rate=float(query.TWS_t_masked.mean()))
    if args.preflight:
        return 0
    log(stage='start', protocol=protocol, variant=args.variant)
    report = {'protocol': protocol, 'variant': args.variant, 'complete': False, 'folds': []}
    for anchor in anchors:
        fit, query, truth = make_replay(train, template, anchor)
        prepared = None
        for seed in seeds:
            key = f'{anchor}/{seed}'
            fold_dir = folder / anchor / str(seed)
            fold_dir.mkdir(parents=True, exist_ok=True)
            score_path = fold_dir / 'scores.json'
            if score_path.exists():
                score = json.loads(score_path.read_text())
                log(stage='fold_resumed', key=key)
            else:
                def fold_log(**row):
                    log(key=key, **row)
                if prepared is None:
                    prepared = prepare(fit, query, states, fold_log)
                train_input, query_input, x_fit, x_query = prepared
                models = fit_heads(x_fit, fit.target.to_numpy(np.float32), fit,
                                   pd.Period(anchor).to_timestamp(), seed, args, fold_dir, fold_log)
                static, recursive = rollout(models, train_input, query_input, x_query, args, fold_dir,
                                             fold_log, truth, query['_offset'].to_numpy())
                score = {'key': key, **score_predictions(query, truth, recursive, month_weights),
                         'static': score_predictions(query, truth, static, month_weights)}
                prediction_table = query[['time', 'lat', 'lon', '_offset', 'TWS_t_masked']].copy()
                prediction_table['target'] = truth
                prediction_table['static'] = static
                prediction_table['recursive'] = recursive
                prediction_table.to_parquet(fold_dir / 'predictions.parquet', index=False)
                atomic_json(score_path, score)
                del models
                gc.collect()
                log(stage='fold_complete', **score)
            report['folds'].append(score)
            atomic_json(folder / 'report.json', report)
        prepared = None
        # Release the previous anchor's large matrices before the next build.
        if 'x_fit' in locals():
            del train_input, query_input, x_fit, x_query
        gc.collect()
    report['complete'] = True
    reference = json.loads(args.reference.read_text()) if args.reference else None
    report['gate'] = promotion_gate(report, reference)
    atomic_json(folder / 'report.json', report)
    log(stage='validation_complete', gate=report['gate'])
    if not args.full or not report['gate']['passed']:
        log(stage='complete', full_generated=False)
        return 0

    # The only path to full-data training is below the complete paired CV gate.
    import baseline
    sample = pd.read_csv(ROOT / 'data' / 'SampleSubmission.csv')
    train_input, query_input, x_fit, x_query = prepare(train, official_template, states, log)
    for seed in seeds:
        full_dir = folder / 'full' / str(seed)
        full_dir.mkdir(parents=True, exist_ok=True)
        models = fit_heads(x_fit, train.target.to_numpy(np.float32), train,
                           official_template.time.min(), seed, args, full_dir, log)
        _, pred = rollout(models, train_input, query_input, x_query, args, full_dir, log)
        submission = baseline.make_submission_frame(official_template, pred, sample)
        if not submission.ID.equals(sample.ID) or not np.isfinite(submission.Target).all():
            raise ValueError('invalid full submission')
        submission.to_csv(full_dir / 'submission.csv', index=False)
        del models
    log(stage='complete', full_generated=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

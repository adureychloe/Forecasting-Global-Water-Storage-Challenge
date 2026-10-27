"""One-fold raw-target depth6 candidate with an early static screen.

No full-data training or submission path. Reuses the verified #258 protocol;
screening on this fold is exploratory, not a final promotion result.
"""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import gpu_sparse_replay as core
from sparse_replay import make_replay, calendar_template, score_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', default='control_20260910')
    p.add_argument('--run-id', required=True)
    p.add_argument('--anchor', default='2007-09')
    p.add_argument('--seed', type=int, default=2027)
    args = p.parse_args()
    for value in (args.source_run, args.run_id):
        if Path(value).name != value or value in ('.', '..'):
            raise ValueError('invalid run name')
    source = core.ROOT / 'results' / 'sparse_replay' / args.source_run
    manifest = json.loads((source / 'manifest.json').read_text())
    protocol = manifest['protocol']
    if manifest['variant'] != 'raw' or protocol['inner_iters'] != 3:
        raise ValueError('requires raw recursion3 reference')
    report = json.loads((source / 'report.json').read_text())
    if not report['complete'] or args.anchor not in protocol['anchors'] or args.seed not in protocol['seeds']:
        raise ValueError('incomplete or missing reference fold')
    for name, expected in manifest['code_sha256'].items():
        if core.digest(core.ROOT / name) != expected:
            raise ValueError(f'changed source code: {name}')
    for name, key in [('Train.csv', 'train_sha256'), ('Test.csv', 'test_sha256')]:
        if core.digest(core.ROOT / 'data' / name) != protocol[key]:
            raise ValueError(f'changed data: {name}')
    fold = source / args.anchor / str(args.seed)
    output = core.ROOT / 'results' / 'capacity_probe' / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    identity = {'source_manifest': manifest, 'source_fold': str(fold), 'depth': 6,
                'runner_sha256': core.digest(Path(__file__)),
                'reference_sha256': core.digest(fold / 'predictions.parquet')}
    mp = output / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != identity:
        raise ValueError('changed run recipe; choose new run-id')
    core.atomic_json(mp, identity)
    def log(**row):
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), **row}
        text = json.dumps(row, allow_nan=False)
        with (output / 'events.jsonl').open('a') as f:
            f.write(text + '\n')
        print(text, flush=True)

    train = pd.read_csv(core.ROOT / 'data' / 'Train.csv', parse_dates=['time'])
    template = pd.read_csv(core.ROOT / 'data' / 'Test.csv', parse_dates=['time'])
    weights = calendar_template(template)['_offset'].value_counts().to_dict()
    states = template.groupby('time', sort=True).TWS_t_masked.mean().ge(.5).to_numpy()
    if protocol['max_cells']:
        cells = train[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon']).head(protocol['max_cells'])
        train = train.merge(cells, on=['lat', 'lon'])
        template = template.merge(cells, on=['lat', 'lon'])
    fit, query, truth = make_replay(train, template, args.anchor)
    reference = pd.read_parquet(fold / 'predictions.parquet')
    keys = ['time', 'lat', 'lon', '_offset', 'TWS_t_masked']
    pd.testing.assert_frame_equal(query[keys], reference[keys])
    np.testing.assert_array_equal(truth, reference.target.to_numpy())
    log(stage='start', depth=6, reference_depth=8, anchor=args.anchor, seed=args.seed,
        iterations=protocol['iterations'], query_rows=len(query))
    train_input, query_input, x_fit, x_query = core.prepare(fit, query, states, log)
    import gpu_catboost_idw_era5_pattern as base
    original_model = base.model
    def depth6_model(seed, iterations=500):
        return original_model(seed, iterations).set_params(depth=6)
    base.model = depth6_model
    recipe = SimpleNamespace(iterations=protocol['iterations'], recency_months=protocol['recency_months'],
        min_head_rows=protocol['min_head_rows'], variant='raw', inner_iters=3)
    models = core.fit_heads(x_fit, fit.target.to_numpy(np.float32), fit,
                           pd.Period(args.anchor).to_timestamp(), args.seed, recipe, output, log)
    if not models or any(m.get_all_params()['depth'] != 6 for m in models.values()):
        raise ValueError('wrong model depth')
    static = core.predict_heads(models, x_query, 'raw')
    static_score = score_predictions(query, truth, static, weights)
    reference_static = score_predictions(query, truth, reference.static.to_numpy(), weights)
    passed = all(static_score[k] <= reference_static[k] - .0005 for k in ('all', 'masked'))
    result = {'complete': False, 'anchor': args.anchor, 'seed': args.seed, 'depth': 6,
              'static': static_score, 'reference_static': reference_static,
              'static_screen_passed': passed, 'full_generated': False,
              'scope': 'single-fold exploration; not online promotion'}
    np.save(output / 'static.npy', static)
    core.atomic_json(output / 'report.json', result)
    log(stage='static_screen', **result)
    del x_fit, train
    gc.collect()
    if passed:
        _, recursive = core.rollout(models, train_input, query_input, x_query, recipe, output,
                                     log, truth, query['_offset'].to_numpy())
        result['recursive'] = score_predictions(query, truth, recursive, weights)
        result['reference_recursive'] = score_predictions(query, truth, reference.recursive.to_numpy(), weights)
        table = reference[keys + ['target']].copy()
        table['static'] = static
        table['recursive'] = recursive
        table.to_parquet(output / 'predictions.parquet', index=False)
    result['complete'] = True
    core.atomic_json(output / 'report.json', result)
    log(stage='complete', **{k:v for k,v in result.items() if k != 'complete'})


if __name__ == '__main__':
    main()

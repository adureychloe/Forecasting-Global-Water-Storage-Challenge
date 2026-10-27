"""Inference-only 2x2 diagnostic reusing an immutable sparse-replay fold."""
import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import gpu_sparse_replay as core
from inference_ablation import spatial_predictions
from sparse_replay import calendar_template, make_replay, score_predictions


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
    output = core.ROOT / 'results' / 'inference_ablation' / args.run_id
    if output.resolve() == source.resolve():
        raise ValueError('output must differ from source')
    manifest = json.loads((source / 'manifest.json').read_text())
    source_report = json.loads((source / 'report.json').read_text())
    protocol = manifest['protocol']
    if not source_report['complete'] or manifest['variant'] != 'raw' or protocol['inner_iters'] != 3:
        raise ValueError('requires completed raw recursion3 baseline')
    if args.anchor not in protocol['anchors'] or args.seed not in protocol['seeds']:
        raise ValueError('fold not in source protocol')
    for name, expected in manifest['code_sha256'].items():
        if core.digest(core.ROOT / name) != expected:
            raise ValueError(f'source code changed: {name}')
    for name, key in [('Train.csv', 'train_sha256'), ('Test.csv', 'test_sha256')]:
        if core.digest(core.ROOT / 'data' / name) != protocol[key]:
            raise ValueError(f'source data changed: {name}')
    fold = source / args.anchor / str(args.seed)
    model_paths = sorted(fold.glob('*.cbm'))
    if {x.stem for x in model_paths} != set(core.BUCKETS):
        raise ValueError('diagnostic requires all five saved heads')
    identity = {'source_manifest': manifest, 'source_fold': str(fold),
                'source_report_sha256': core.digest(source / 'report.json'),
                'models': {x.name: core.digest(x) for x in model_paths},
                'source_predictions_sha256': core.digest(fold / 'predictions.parquet'),
                'code': {x: core.digest(core.ROOT / x) for x in
                         ['gpu_inference_ablation.py', 'inference_ablation.py']}}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError('run-id contains a different experiment')
    core.atomic_json(path, identity)

    def log(**row):
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), **row}
        text = json.dumps(row, allow_nan=False)
        with (output / 'events.jsonl').open('a') as f:
            f.write(text + '\n')
        print(text, flush=True)

    from catboost import CatBoostRegressor
    import neighborhood_tws_features as nb
    models = {x.stem: CatBoostRegressor().load_model(str(x)) for x in model_paths}
    log(stage='models_loaded', count=len(models), retraining=False)
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
    pd.testing.assert_frame_equal(query[keys].reset_index(drop=True), reference[keys].reset_index(drop=True))
    np.testing.assert_array_equal(truth, reference.target.to_numpy())
    train_input, query_input, x_fit, x_query = core.prepare(fit, query, states, log)
    del x_fit, train
    gc.collect()
    pred = lambda x: core.predict_heads(models, x, 'raw')
    static = pred(x_query)
    np.testing.assert_allclose(static, reference.static.to_numpy(), rtol=1e-6, atol=1e-6)
    log(stage='static_reference_reproduced', rows=len(static))
    spatial = np.empty(len(query), dtype=np.float32)
    for month, current in query_input.groupby('time', sort=True):
        pos = current.index.to_numpy()
        spatial[pos] = spatial_predictions(current, x_query.loc[pos], pred,
            lambda raw, x, values: core.clean(nb.overlay_pred_neighborhood(raw, x, values)), 3)
        if month == query_input.time.min():
            np.testing.assert_allclose(spatial[pos], reference.recursive.to_numpy()[pos], rtol=1e-6, atol=1e-6)
            log(stage='first_month_recursive_reference_reproduced')
        log(stage='spatial_only_month', month=str(month.date()), rows=len(pos))
    table = reference[keys + ['target']].copy()
    table['direct'] = static
    table['spatial_only'] = spatial
    table['feedback_and_spatial'] = reference.recursive.to_numpy()
    scores = {name: score_predictions(query, truth, table[name].to_numpy(), weights)
              for name in ['direct', 'spatial_only', 'feedback_and_spatial']}
    core.atomic_json(output / 'report.json', {'complete': False, 'scores': scores})
    table.to_parquet(output / 'predictions_partial.parquet', index=False)
    log(stage='spatial_only_complete', scores=scores)
    feedback_dir = output / 'feedback_only'
    feedback_dir.mkdir(exist_ok=True)
    _, feedback = core.rollout(models, train_input, query_input, x_query,
                              SimpleNamespace(variant='raw', inner_iters=1), feedback_dir,
                              log, truth, query['_offset'].to_numpy())
    table['feedback_only'] = feedback
    scores['feedback_only'] = score_predictions(query, truth, feedback, weights)
    table.to_parquet(output / 'predictions.parquet', index=False)
    core.atomic_json(output / 'report.json', {'complete': True, 'source_run': args.source_run,
        'anchor': args.anchor, 'seed': args.seed, 'rows': len(query), 'scores': scores,
        'scope': 'one-fold diagnostic; no promotion or test submission'})
    log(stage='complete', scores=scores, retraining=False, full_generated=False)


if __name__ == '__main__':
    main()

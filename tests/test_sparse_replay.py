"""Regression tests for calendar coverage, information boundaries and promotion."""
import unittest
import numpy as np
import pandas as pd

import sparse_replay as replay


def fixture():
    train = pd.DataFrame({
        'time': pd.date_range('2010-01-01', '2014-12-01', freq='MS'),
        'lat': 1., 'lon': 2., 'TWS_t': np.arange(60.),
        'target': np.arange(60.) + 1,
    })
    test = pd.DataFrame({
        'time': pd.to_datetime(['2020-01-01', '2020-05-01', '2020-06-01', '2020-07-01']),
        'lat': 1., 'lon': 2., 'TWS_t_masked': [False, False, True, True],
    })
    return train, test


class SparseReplayTest(unittest.TestCase):
    def test_full_fit_cannot_start_without_reference_or_with_smoke_subset(self):
        from gpu_sparse_replay import main
        with self.assertRaisesRegex(SystemExit, 'requires a reference'):
            main(['--run-id', 'should_not_run', '--full'])
        with self.assertRaisesRegex(SystemExit, 'requires a reference'):
            main(['--run-id', 'should_not_run', '--full', '--reference', 'anything.json', '--max-cells', '32'])

    def test_preserves_sparse_calendar_and_excludes_intermediate_observations(self):
        train, template = fixture()
        fit, query, truth = replay.make_replay(train, template, '2011-01')
        self.assertEqual(query.time.dt.strftime('%Y-%m').tolist(),
                         ['2011-01', '2011-05', '2011-06', '2011-07'])
        self.assertLess(fit.time.max(), pd.Timestamp('2011-01-01'))
        self.assertNotIn('target', query)
        np.testing.assert_array_equal(truth, [13., 17., 18., 19.])
        self.assertTrue(query.loc[2:, 'TWS_t'].isna().all())
        self.assertEqual(query.loc[1, 'TWS_t'], 16.)

    def test_preserves_cell_level_mask_exceptions(self):
        train, template = fixture()
        train = pd.concat([train, train.assign(lon=3.)], ignore_index=True)
        other = template.assign(lon=3.)
        other.loc[2, 'TWS_t_masked'] = False
        template = pd.concat([template, other], ignore_index=True)
        _, q, _ = replay.make_replay(train, template, '2011-01')
        month = q.loc[q.time == '2011-06-01']
        self.assertEqual(int(month.TWS_t.isna().sum()), 1)

    def test_missing_calendar_month_is_rejected_not_silently_compressed(self):
        train, template = fixture()
        train = train.loc[train.time != '2011-05-01']
        with self.assertRaisesRegex(ValueError, 'month'):
            replay.make_replay(train, template, '2011-01')

    def test_legacy_walk_is_causal_and_matches_historical_feedback_convention(self):
        train, template = fixture()
        fit, q, _ = replay.make_replay(train, template, '2011-01')

        def predict(history, month):
            self.assertNotIn('target', history)
            self.assertTrue((history.time < month.time.min()).all())
            # Stand-in forecast with an explicit dependence on the latest input.
            return np.repeat(history.TWS_t.dropna().iloc[-1] + 1., len(month))

        p = replay.walk_legacy(fit.drop(columns='target'), q, predict)
        np.testing.assert_array_equal(p, [12., 13., 17., 18.])
        modified = q.copy()
        modified.loc[1, 'TWS_t'] = 999.
        p2 = replay.walk_legacy(fit.drop(columns='target'), modified, predict)
        self.assertEqual(p[0], p2[0])
        self.assertNotEqual(p[2], p2[2])
        pd.testing.assert_frame_equal(q, replay.make_replay(train, template, '2011-01')[1])

    def test_aligned_walk_writes_forecast_into_next_month_not_current(self):
        from aligned_feedback import walk_aligned
        train, template = fixture()
        fit, q, _ = replay.make_replay(train, template, '2011-01')
        seen = []

        def predict(history, month):
            self.assertNotIn('target', history)
            self.assertTrue((history.time < month.time.min()).all())
            seen.append(float(pd.to_numeric(month.TWS_t, errors='coerce').isna().mean()))
            return np.repeat(history.TWS_t.dropna().iloc[-1] + 1., len(month))

        p = walk_aligned(fit.drop(columns='target'), q, predict)
        # 2011-01/05 are isolated or observed; 2011-06/07 are consecutive masked.
        # Aligned fill applies the previous forecast onto the next month before predict.
        np.testing.assert_array_equal(p[:2], [12., 13.])
        self.assertEqual(p[2], 17.)
        self.assertEqual(p[3], 14.)
        self.assertLess(p[3], 18.)
        hist = fit.drop(columns='target')
        # Query frame is not mutated.
        pd.testing.assert_frame_equal(q, replay.make_replay(train, template, '2011-01')[1])

    def test_metric_uses_test_month_weights_not_mean_of_rmse(self):
        q = pd.DataFrame({'_offset': [0, 0, 4], 'TWS_t_masked': [False, False, True]})
        scores = replay.score_predictions(q, np.zeros(3), np.array([0., 0., 2.]), {0: 1, 4: 3})
        self.assertAlmostEqual(scores['all'], np.sqrt(3.))
        self.assertEqual(scores['masked'], 2.)

    def test_gate_rejects_partial_protocol_mismatch_nonfinite_and_regression(self):
        protocol = {'v': 1, 'anchors': ['a', 'b'], 'seeds': [77]}
        ref = {'protocol': protocol, 'complete': True, 'folds': [
            {'key': 'a/77', 'all': .7, 'masked': .8},
            {'key': 'b/77', 'all': .6, 'masked': .7}]}
        good = {'protocol': protocol, 'complete': True, 'folds': [
            {'key': 'a/77', 'all': .695, 'masked': .795},
            {'key': 'b/77', 'all': .595, 'masked': .695}]}
        self.assertTrue(replay.promotion_gate(good, ref)['passed'])
        self.assertFalse(replay.promotion_gate(good, None)['passed'])
        self.assertFalse(replay.promotion_gate({**good, 'complete': False}, ref)['passed'])
        self.assertFalse(replay.promotion_gate({**good, 'protocol': {'v': 2}}, ref)['passed'])
        self.assertFalse(replay.promotion_gate({**good, 'folds': good['folds'][:1]}, ref)['passed'])
        self.assertFalse(replay.promotion_gate(
            {**good, 'folds': good['folds'][:1]},
            {**ref, 'folds': ref['folds'][:1]})['passed'])
        bad = {**good, 'folds': [dict(good['folds'][0], all=float('nan')), good['folds'][1]]}
        self.assertFalse(replay.promotion_gate(bad, ref)['passed'])
        bad['folds'][0]['all'] = .75
        self.assertFalse(replay.promotion_gate(bad, ref)['passed'])


if __name__ == '__main__':
    unittest.main()

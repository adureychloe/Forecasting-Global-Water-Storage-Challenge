import unittest
import numpy as np
import pandas as pd
from inference_ablation import spatial_predictions


class SpatialAblationTest(unittest.TestCase):
    def test_iterations_use_predictions_but_do_not_mutate_static_inputs(self):
        raw = pd.DataFrame({'TWS_t_masked': [True, False]})
        features = pd.DataFrame({'value': [1., 4.], 'TWS_t_age': [2., 0.]})
        original = features.copy()
        def predict(x):
            return x.value.to_numpy() + 1
        def overlay(r, x, p):
            x = x.copy()
            x.loc[r.TWS_t_masked, 'value'] = p[r.TWS_t_masked]
            return x
        np.testing.assert_array_equal(spatial_predictions(raw, features, predict, overlay, 1), [2, 5])
        np.testing.assert_array_equal(spatial_predictions(raw, features, predict, overlay, 3), [4, 5])
        pd.testing.assert_frame_equal(features, original)

    def test_invalid_prediction_and_changed_routing_are_rejected(self):
        raw = pd.DataFrame({'TWS_t_masked': [True]})
        x = pd.DataFrame({'TWS_t_age': [2.]})
        with self.assertRaisesRegex(ValueError, 'prediction'):
            spatial_predictions(raw, x, lambda _: np.array([np.nan]), lambda r,x,p:x, 1)
        def overlay(r, x, p):
            return x.assign(TWS_t_age=0.)
        with self.assertRaisesRegex(ValueError, 'routing'):
            spatial_predictions(raw, x, lambda _: np.array([1.]), overlay, 3)


if __name__ == '__main__':
    unittest.main()

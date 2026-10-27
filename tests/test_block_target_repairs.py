import unittest

import numpy as np
import pandas as pd

import block_centered_target
import block_robust_normalized_target


class BlockTargetRepairTest(unittest.TestCase):
    def setUp(self):
        self.features = pd.DataFrame(
            {
                "target_block_roll24_mean": [2.0, 3.0, 4.0],
                "target_block_roll24_std": [0.0, 0.05, 0.5],
                "target_block_last": [2.1, 3.1, 4.1],
            }
        )
        self.target = np.array([2.5, 3.5, 4.5], dtype=np.float32)

    def test_robust_normalization_falls_back_for_short_history(self):
        center, scale = block_robust_normalized_target.center_scale(self.features)
        np.testing.assert_allclose(center, [0.0, 0.0, 4.0])
        np.testing.assert_allclose(scale, [1.0, 1.0, 0.5])
        transformed = block_robust_normalized_target.transform_targets(
            self.target, self.features
        )
        restored = block_robust_normalized_target.inverse_predictions(
            transformed, self.features
        )
        np.testing.assert_allclose(restored, self.target)

    def test_center_only_transform_is_invertible(self):
        transformed = block_centered_target.transform_targets(self.target, self.features)
        restored = block_centered_target.inverse_predictions(transformed, self.features)
        np.testing.assert_allclose(restored, self.target)
        np.testing.assert_allclose(transformed, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()

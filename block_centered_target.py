"""Target residuals around a strictly block-frozen coordinate center."""

import numpy as np


def center(features):
    return features["target_block_roll24_mean"].to_numpy(dtype=np.float32)


def transform_targets(target, features):
    return (np.asarray(target, dtype=np.float32) - center(features)).astype(np.float32)


def inverse_predictions(prediction, features):
    return (np.asarray(prediction, dtype=np.float32) + center(features)).astype(np.float32)

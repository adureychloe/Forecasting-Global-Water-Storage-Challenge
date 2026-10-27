"""Block-frozen normalization with identity fallback for short histories."""

import numpy as np


def center_scale(features):
    center = features["target_block_roll24_mean"].to_numpy(dtype=np.float32)
    scale = features["target_block_roll24_std"].to_numpy(dtype=np.float32)
    stable = scale >= 0.10
    center = np.where(stable, center, 0.0).astype(np.float32)
    scale = np.where(stable, np.clip(scale, 0.10, 2.0), 1.0).astype(np.float32)
    return center, scale


def transform_targets(target, features):
    center, scale = center_scale(features)
    return ((np.asarray(target, dtype=np.float32) - center) / scale).astype(np.float32)


def inverse_predictions(prediction, features):
    center, scale = center_scale(features)
    return (np.asarray(prediction, dtype=np.float32) * scale + center).astype(np.float32)

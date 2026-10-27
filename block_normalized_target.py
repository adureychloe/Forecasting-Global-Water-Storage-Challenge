"""Target-space transforms using strictly block-frozen coordinate statistics."""

import numpy as np


def center_scale(features):
    center = features["target_block_roll24_mean"].to_numpy(dtype=np.float32)
    scale = features["target_block_roll24_std"].to_numpy(dtype=np.float32)
    last = features["target_block_last"].to_numpy(dtype=np.float32)
    available = (np.abs(center) + np.abs(last) + np.abs(scale)) > 1e-7
    center = np.where(available, center, 0.0).astype(np.float32)
    scale = np.where(available, np.clip(scale, 0.08, 2.0), 1.0).astype(np.float32)
    return center, scale


def transform_targets(target, features):
    center, scale = center_scale(features)
    return ((np.asarray(target, dtype=np.float32) - center) / scale).astype(np.float32)


def inverse_predictions(prediction, features):
    center, scale = center_scale(features)
    return (np.asarray(prediction, dtype=np.float32) * scale + center).astype(np.float32)

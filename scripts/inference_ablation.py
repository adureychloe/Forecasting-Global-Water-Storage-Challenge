"""Within-month spatial iteration without feeding predictions into later months."""
import numpy as np


def spatial_predictions(raw, features, predict, overlay, iterations=3):
    if iterations < 1:
        raise ValueError('iterations must be positive')
    features = features.copy().reset_index(drop=True)
    raw = raw.reset_index(drop=True)
    age = features['TWS_t_age'].to_numpy().copy()
    for iteration in range(iterations):
        if iteration:
            features = overlay(raw, features, pred)
            if not np.array_equal(features['TWS_t_age'].to_numpy(), age):
                raise ValueError('spatial overlay changed age routing')
        pred = np.asarray(predict(features), dtype=np.float32)
        if pred.shape != (len(raw),) or not np.isfinite(pred).all():
            raise ValueError('invalid spatial prediction')
    return pred

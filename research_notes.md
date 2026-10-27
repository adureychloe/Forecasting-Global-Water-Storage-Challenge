# Similar Kaggle Forecasting Research

## 2026-08-24: Panel and multi-horizon solutions

### M5 Forecasting Accuracy, 4th place

Source: `monsaraida/kaggle-m5-forecasting-accuracy-4th-place`.

- Used multiple rolling-origin folds rather than random validation.
- Trained separate regional/store models.
- Compared weekly horizon heads (`7, 14, 21, 28`), one monthly head, and day-by-day heads.
- Mapping here: preserve chronological folds, test query-length block alignment, and compare shared versus explicit horizon heads.

### Corporacion Favorita, 12th place

Source: `antklen/kaggle_favorita-grocery-sales`.

- Combined one general LightGBM trained across all lags with separate per-lag LightGBM and neural models.
- Mapping here: retain a shared model as the variance anchor, then add direct horizon-specific residual models only when OOF evidence supports them.

### Web Traffic Forecasting, 11th place

Source: `louis925/kaggle-web-traffic-time-series-forecasting`.

- Used robust entity-level center and scale before CNN training.
- Split entities by scale, predicted a full future sequence, and median-ensembled five fits.
- Mapping here: use strictly past, block-frozen coordinate center/scale; predict normalized TWS residuals and reconstruct the absolute target. A block sequence-to-sequence model is a later neural variant.

### Predict Future Sales, Top 15 report

Source: `jukiewiczm/kaggle-predict-future-sales`.

- Used time-aware rolling validation, entity embeddings, lagged aggregates, and RNN sequence models.
- Mapping here: categorical coordinate CTR and TCN/RNN routes are already queued; evaluate them against identical chronological folds.

## Prioritized transfers

1. Test query-span-aligned training blocks: fold0 spans 12 calendar months, sparse fold1 spans 8, and the 18 submitted Test dates span 40; derive each block length from its query while keeping statistics frozen before query start.
2. Predict block-frozen coordinate-standardized residuals and invert using fit-only center/scale.
3. Train a mask-pattern-aligned block sequence model with shared multi-horizon output and robust coordinate normalization.

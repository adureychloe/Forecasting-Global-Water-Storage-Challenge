# Sparse-calendar recursive validation

Experiment #258 establishes a new validation control for the current CatBoost recipe. It does not change the confirmed online leader (RMSE 0.7139) or submit anything online.

## Why this replaces the old pattern score

The official Test contains 18 observed dates spanning 40 calendar months. The old pattern runner assigned month masks to `anchor + i`, which compressed the dates. Its validation also used direct predictions while submission inference used recursive feedback and three spatial iterations. Therefore old ordinary/pattern scores around 0.614/0.596 cannot be compared directly with this protocol.

The new evaluator maps exact offsets `0,4,5,6,9,10,11,12,15,16,17,18,19,20,21,34,38,39` and individual cell masks onto historical dates. It excludes all intervening validation-period observations. Missing historical cells are intersected and reported; missing entire requested months are rejected. Query labels are kept outside inference inputs. Supervised statistics use only the fit period.

## Control recipe and interpretation

- Anchors: 2007-09, 2008-06, 2009-01; seeds: 2027, 77.
- Five age heads, CatBoost GPU Ordered, depth 8, learning rate 0.04, l2 5, fixed 500 iterations. No outer validation early stopping.
- Fit recency: 68 calendar months before each anchor; IDW, ERA5 and block-frozen target statistics; coordinate predictors stripped.
- Predictions: static diagnostic plus legacy monthly feedback and three spatial iterations.
- Overall and masked RMSE: weight month MSE by official Test month row counts, then take square root. Paired-fold aggregate uses equal-fold pooled MSE.
- The three historical windows overlap, so they measure temporal robustness but are not six independent statistical replications.

The legacy feedback deliberately remains unchanged to establish a reference: the old code writes forecast(t+1) back into the missing row at t. The target was verified to equal TWS at the next calendar month. A separately evaluated, time-aligned feedback mode is the next model experiment; #258 does not claim that semantic issue is repaired.

Historical validation coverage before optional smoke subsetting:

| Anchor | Fit rows | Query rows | Test cell/date coverage |
|---|---:|---:|---:|
| 2007-09 | 921,011 | 280,755 | 99.9267% |
| 2008-06 | 1,061,686 | 279,490 | 99.4764% |
| 2009-01 | 1,170,802 | 279,339 | 99.4227% |

## Execution and recovery

Remote code snapshot: `/workspace/water/sparse_replay_code_20260910`, with symlinks to existing data/results. Python: `/venv/main/bin/python`.

Supervisor services:

- `water_gpu_sparse_replay_smoke`: 32 cells, 3 iterations, one anchor/seed; run ID `smoke_v2_20260910`.
- `water_gpu_sparse_replay_control`: all cells, all six anchor/seed pairs; run ID `control_20260910`.

The services are explicitly started, have `autostart=false` and `autorestart=false`, and exit after completion. Supervisor stops the entire process group. They do not keep a sleeping process alive. Server shutdown is not automatic.

```bash
supervisorctl status water_gpu_sparse_replay_control
tail -n 12 /workspace/water/results/sparse_replay/control_20260910/events.jsonl
```

Each run saves its protocol, data/code SHA256 manifest, model heads, monthly checkpoints, OOF Parquet, per-fold scores and report under `/workspace/water/results/sparse_replay/<run-id>/`. Resume requires the same manifest. Use a new run ID for code/recipe changes. Do not edit the running snapshot.

## Promotion gate

Default execution is CV only. `--full` requires a complete comparable reference report and all cells. Both reports must contain all planned anchor/seed pairs, with matching protocols, no duplicate pairs, and finite overall/masked scores. Overall and masked pooled RMSE must each improve by at least 0.0005, with no paired overall/masked regression over 0.002. Failing the gate ends the job without full training or Test generation. The baseline has no reference and is expected to finish with `full_generated=false`.

## Verification

Local command `python -m unittest test_sparse_replay test_sparse_features test_baseline test_block_target_repairs -v`: 16 tests passed. Remote pandas 3 environment: eight replay/feature tests passed. Tests cover exact dates, cell masks, information boundaries, metric weighting, missing planned folds, full-fit entry restrictions and query-only neighborhood equivalence. Real GPU smoke additionally checks first-month complete feature equivalence and all 18 monthly checkpoint/OOF outputs. Small-panel smoke scores are functional diagnostics, not model-quality estimates.

Final-code smoke `smoke_v2_20260910` completed at 2026-09-10 13:36:58 CST. Its downloaded artifacts were checked locally: five saved heads, 18 monthly checkpoints, 576 finite OOF rows, 18 exact dates, complete report and `full_generated=false`. All 17 code hashes match local files. Control started around 13:38 CST; at 13:41 it was RUNNING in first-anchor feature preparation, with no full CV scores yet.

Control completed at2026-09-10 20:07:47 CST after about6h30m; final report/events downloaded and validated (6folds,108monthly steps). Equal-fold pooled RMSE: overall recursive0.542328/static0.526434, masked0.577134/0.554980, observed0.465418/0.464460. All six paired runs favor static on historical replay. This does not establish an online gain. Full generation was skipped by design; service exited, instance remains on.

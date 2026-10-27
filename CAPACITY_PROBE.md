# Depth6 optimization probe #260

This is an actual GPU model candidate, not a further validation-only diagnostic. It changes only CatBoost depth8 to depth6 in the current raw-target IDW+ERA5+block-frozen recipe. Hypothesis: reducing capacity may improve held-out predictions and reduce sensitivity to recursive inputs. This is unproven until measured.

The first screen uses anchor2007-09, seed2027, fixed500iterations and68month recency. Source code/data hashes and OOF identities must match completed #258. Saved candidate models are checked to have depth6. Compare overall and masked static RMSE with the matching depth8 source, requiring at least0.0005 improvement in each. Only passing candidates proceed to the slower recursive3 validation. A failed static screen exits normally. This is a speed-oriented heuristic that can miss candidates whose benefit appears only during recursion; it is not a final promotion gate.

No full/Test generation is implemented here. An improved single fold must be confirmed on further windows before becoming a submission candidate. Existing 0.7139 online leader and source snapshots are preserved. This experiment does not use centering or normalization and does not claim to explain the earlier centered0.79 result.

Service: water_gpu_capacity_probe. Code: /workspace/water/sparse_replay_code_20260910/gpu_capacity_probe.py. Results: /workspace/water/results/capacity_probe/depth6_20260910/. New entry/config only; prior source modules and running #259 files are unchanged. Small-panel source smoke verifies actual depth6 GPU training, scoring and screening before full-panel launch.

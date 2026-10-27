## Status update — 2026-09-13

#277 extra LGBM seed123 2.5% scored **LB 0.6955** (tie). Persist a3 w15 0.6965. Seed2027 2.5% 0.6956. Do not spend extra LGBM 9001 / persist w25 / extra CatBoost. **2 slots left.** No agent submission.

## Status update — 2026-09-12

#272 XGB aligned seed77 30% scored **LB 0.6957**. w200 **0.6970** (−0.0057), w300 **0.6957** (−0.0013). Next: remade `submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w200.csv` (20% into 0.6957). Do not jump to 50%. Never 75%. 7 slots left. No agent submission.

## Status update — 2026-09-11

#269 LGBM aligned seed77 2.5% scored **LB 0.7076** (new best). Next probe: `submission_onlineleader_idw_era5_xgbaligned_seed2027_raw_w025.csv` (remade into 0.7076). Do not spend LGBM aligned 5/10%. GPU #267 XGB seed77 still running. No agent submission.

## Status update — 2026-09-10

#258 completed; #259 started21:05 CST as a single-window inference diagnostic reusing saved models. It compares direct/spatial-only/feedback-only/both to isolate the source of the historical replay difference. No new submission file or candidate promotion; keep0.7139. See INFERENCE_ABLATION.md for current service/results paths.

User now reports **LB approximately 0.79** for repaired `submission_catboost_horizon_idw_era5_pattern_centered_seed77_pattern_raw.csv` (#256). Local and remote file hashes still match exactly; this is model-quality failure, not changed predictions during download repair. Keep #256 rejected; file integrity does not imply promotion. The standalone centered output is not the 0.7139 cumulative ensemble.

Current-stack center-only (#256) and robust normalization (#257) are rejected. Keep LB 0.7139; no new submission recommended. Robust seed77 test inference was stopped after both seeds failed CV. Older queued/running entries below are historical.

14:11 CST integrity repair: centered seed77 `pattern_raw.csv` was an incomplete local download (119912/280961 rows), causing the reported missing-ID rejection. Replaced from the complete server original after hash/ID/finite-value checks; all313 active CSVs now have passing integrity checks (312 unchanged files passed the audit, this file passed after repair). #256's model-quality rejection is unchanged.

#258 started at 13:38 CST: the current CatBoost recipe is being re-evaluated with the true 40-calendar-month / 18-date Test pattern and legacy recursive inference over three anchors and two seeds. Small GPU smoke and regression tests passed. This run is CV-only and establishes a reference; its scores must not be compared directly with the old compressed pattern scores. See SPARSE_REPLAY.md.

# Corrected Candidate Shortlist

Submission directory was cleaned on 2026-08-24. Active files remain in `submission/`; older or lower-priority CSV candidates were moved (not deleted) to `submission_archive_20260824/`.

## Online Leader

The current confirmed online leader is remade `submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w200.csv` with LB RMSE **0.6955** (Experiment #273, 20% XGB77 into 0.6957). Prior w300 **0.6957**.

Local neighbourhood routing (Experiment #212), no remote GPU: blend only masked rows with age>=3 toward a 500 km last-observed neighbour persist, feeding the blend forward recursively. Two-fold LightGBM proxy improved 0.677303 → **0.675637** at weight 0.15.

GPU IDW + ERA5 family (#226) remains the comparable static proxy leader at **0.622694** (seed2027). Recursive #230 training proxy seed77 **0.614575** / seed2027 **0.614801**. Spatial-iteration #232 completed; training proxy seed77 **0.613860** / seed2027 **0.614622**. Spatialiter5 #236 ordinary two-fold matches #232; test vs 3-iter RMS **0.0166 / 0.0165**. Recency 2012+ #237 ordinary two-fold matches #232; test vs spatialiter RMS **0.303 / 0.304**. Teacher two-pass #238 ordinary two-fold matches #232; test vs spatialiter RMS **1.589 / 1.547** (large shift). XGB #241 ordinary seed77 **0.616464** / seed2027 **0.616383**; test vs CatBoost pattern RMS **0.213 / 0.210**. Gap-recency #242 ordinary matches #232; test vs spatialiter RMS **0.000 / 0.000** (DRY duplicate). Extra 2010+ seeds #243 ordinary seed123 **0.614960** / seed9001 **0.614222**; test vs pattern seed77 RMS **0.120 / 0.117** (ensemble-only). Extra XGB #245 ordinary seed123 **0.615789** / seed9001 **0.615522**; test vs xgbpattern seed77 RMS **0.127 / 0.127** (ensemble-only, not clearly better). LightGBM #247 ordinary seed77 **0.615283** / seed2027 **0.615616**; pattern **0.602415 / 0.599979**; test vs xgbpattern RMS **0.151 / 0.148** (third family, transferred at 2.5% to 0.7143). Extra LGBM #249 ordinary seed123 **0.614969** / seed9001 **0.614530**; vs lgbm77 RMS **0.120 / 0.120**; ens2 **0.103** (ensemble-only).

1. remade `submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w200.csv` (current online leader, 0.6955)
2. Seed2027 2.5% scored 0.6956 — rejected. Do not spend seed2027 5/10% or seed77 `w500`. Never 75%.

Local persist-routing files remain optional and lower priority:

- `submission_onlineleader_w100_nbrec_a3_w15.csv`
- `submission_onlineleader_w100_nbrec_a3_w25.csv`

Do not submit the `a7` variants (no test rows have TWS age>=7) or the all-masked `a0` files (fold1 collapses). Local `submission_local_lgbm_nocoord_*.csv` files are diagnostic only.

| File | Mix | Public LB |
|---|---|---|
| old block-frozen `..._b08_w100.csv` | — | 0.741526 |
| neighbourhood `b03_w025` | 2.5% × 30% new | 0.7413 |
| neighbourhood `b03_w100` | 10% × 30% new | 0.7407 |
| IDW `b03_w025` | 2.5% × 30% new | 0.7412 |
| IDW `b03_w100` | 10% × 30% new | 0.7404 |
| IDW `raw_w100` | 10% × 100% new | 0.738 |
| IDW `raw_w500` | 50% IDW into old `w100` | 0.7352 |
| IDW `raw_w750` | 75% IDW | 0.743 (rejected) |
| ERA5 `raw_w025` | 2.5% ERA5 into IDW 50% | 0.7345 |
| ERA5 `raw_w100` | 10% ERA5 into IDW 50% | 0.7329 |
| ERA5 `raw_w300` | 30% ERA5 into IDW 50% | 0.7311 |
| recursive `raw_w100` | 10% recursive into ERA5 30% | 0.7232 |
| spatialiter `seed77_raw_w100` | 10% spatialiter into recursive 10% | 0.7179 |
| pattern `seed77_raw_w100` | 10% 2010+ recency into spatialiter 10% | 0.7173 |
| recency2012 `seed77_raw_w100` | 10% 2012+ into pattern 10% | 0.7192 (rejected) |
| xgbpattern `seed77_raw_w100` | 10% XGB 2010+ into CatBoost pattern 10% | 0.715 |
| patternextra ens4 `raw_w025` | 2.5% four-seed CatBoost 2010+ into XGB 10% | 0.7153 (rejected) |
| xgbextra ens2 `raw_w025` | 2.5% extra XGB 123+9001 into XGB 10% | 0.7147 |
| lgbmpattern `seed77_raw_w025` | 2.5% LightGBM seed77 into extra-XGB ens2 | 0.7143 |
| lgbmextra ens2 `raw_w025` | 2.5% extra LGBM 123+9001 into LGBM seed77 | 0.7140 |
| lgbmpattern `seed2027_raw_w025` | 2.5% LightGBM seed2027 into LGBM seed77 | 0.7139 |
| anomalag `seed2027_raw_w025` | 2.5% CatBoost lagged ERA5 anom into LGBM seed2027 | 0.7142 (rejected) |
| patternstatic `seed77_raw_w025` | 2.5% CatBoost 2010+ static (no month-walk) into LGBM seed2027 | 0.7139 (tie) |
| aligned `seed77_raw_w025` | 2.5% CatBoost 2010+ aligned walk into LGBM seed2027 | 0.7121 |
| aligned `seed2027_raw_w025` | 2.5% CatBoost aligned seed2027 into aligned seed77 | 0.7105 |
| lgbmaligned `seed2027_raw_w025` | 2.5% LGBM 2010+ aligned walk into CatBoost aligned seed2027 | 0.7090 |
| lgbmaligned `seed77_raw_w025` | 2.5% LGBM aligned seed77 into LGBM aligned seed2027 | 0.7076 |
| xgbaligned `seed77_raw_w100` | 10% XGB aligned seed77 into LGBM aligned seed77 2.5% | 0.7027 |
| xgbaligned `seed77_raw_w200` (into 0.7027) | 20% XGB77 into 0.7027 | 0.6970 |
| xgbaligned `seed77_raw_w300` | 30% XGB77 into 0.7027 | 0.6957 |
| **xgbaligned remade `seed77_raw_w200`** | **20% XGB77 into 0.6957** | **0.6955** |
| xgbaligned `seed2027_raw_w025` | 2.5% XGB aligned seed2027 into 0.6955 | 0.6956 (rejected) |
| persist `xgbaligned77_w200_nbrec_a3_w15` | neighbourhood persist age>=3 w0.15 on 0.6955 | 0.6965 (rejected) |
| lgbmaligned extra `seed123_raw_w025` | 2.5% extra LGBM seed123 into 0.6955 | 0.6955 (tie) |

## Next Manual Submissions

2 slots left. Extra LGBM seed123 2.5% tied 0.6955.

Next if asked: `submission_onlineleader_idw_era5_xgbaligned_seed77_raw_w250.csv` — 25% XGB77 into the 0.6957 parent, same construction as winning remade w200 (20%). Small step; seed77 ramp was nearly saturated.

Do not spend: persist w25, extra LGBM 9001, extra CatBoost, seed77 w500, seed2027 5/10%, 75%. No agent submission.

All listed files have 280,961 rows, exact SampleSubmission ID order, and finite predictions. No online submission is made by the agent.

## New Proxy Leader: Block-Frozen Target History

Query-aligned 12-month block-frozen last/rolling12/rolling24 target statistics improved every fold and both seeds without validation/test labels or future TWS. The two-fold means are **0.639744590** for seed2027 and `0.641194192` for seed77, versus `0.650313876`/`0.649573589` for the simpler past-target statistics.

Lower-priority 2.5% probes, only after the weight ramp saturates:

- `submission_onlineleader_targetstats_blockfrozen_ens4_b08_w025.csv`
- `submission_onlineleader_targetstats_blockfrozen_ens2_b08_w025.csv`
- `submission_onlineleader_targetstats_blockfrozen_seed2027_b07_w025.csv`
- `submission_onlineleader_xgb_blockfrozen_seed27182_b075_w025.csv`
- `submission_onlineleader_lgbm_blockfrozen_seed9001_b075_w025.csv`

Block40 remains last because exact sparse replay improved overall RMSE by only `0.000499874` and was mixed across anchors.

Categorical location CTRs completed at seed77 `0.650168492` and seed2027 `0.651014623`, both behind the ordinary and block-frozen leaders. They do not enter the primary manual order. Optional diversity probes, only after the five block-frozen candidates above, are `submission_onlineleader_loccat_seed77_b07_w025.csv` and `submission_onlineleader_loccat_ens2_b07_w025.csv`; do not test their `w050` variants unless a 2.5% probe improves LB. Adding CTRs to block-frozen history also failed across both seeds, so that combined family is DRY and has no recommended submission.

The fixed 5-NN graph TCN improved the original sequence proxy from `0.653780593` to `0.648696594`. It remains below block-frozen trees, so `submission_torch_graph_tcn_sequence_onlineleader_w05.csv` is a tail diversity candidate only, after the five primary submissions and before any 10% graph weight.

Three-region block-frozen heads produced mixed folds and a weaker overall proxy (`0.641061282`). Keep `submission_onlineleader_blockregion3_seed2027_b07_w025.csv` only as a lower-priority spatial diversity probe; do not test its 5% version unless 2.5% improves LB.

Block-frozen memory transferred strongly to XGBoost (`0.642175137`, a `0.010291478` gain over ordinary enhanced XGBoost). It remains below CatBoost but is algorithmically distinct; `submission_onlineleader_xgb_blockfrozen_seed27182_b075_w025.csv` is the preferred tree-diversity probe after the five primary block-frozen submissions.

LightGBM confirms the same transfer (`0.642518992`, improving its ordinary variant by `0.008872911`). `submission_onlineleader_lgbm_blockfrozen_seed9001_b075_w025.csv` is a secondary third-family probe after the XGBoost candidate; test 5% only if 2.5% improves LB.

## Previous Proxy Leader: Past Target Statistics

M5-inspired fit-only coordinate statistics produced the first material local improvement after the Enhanced Ordered control. Training rows use expanding statistics shifted by one row; validation and test rows map aggregates built only from the corresponding fit period. No validation labels, test labels, future TWS, or negative shifts are used.

| Family | Two-fold proxy | Priority candidates |
|---|---:|---|
| Target stats seed77 | **0.649573589** | `submission_onlineleader_targetstats_seed77_b07_w025.csv`, then `submission_onlineleader_targetstats_seed77_b07_w050.csv` |
| Target stats seed2027 | 0.650313876 | `submission_onlineleader_targetstats_seed2027_b07_w025.csv` |
| Two-seed aligned ensemble | component evidence above | `submission_onlineleader_targetstats_ens2_b07_w025.csv`, then `submission_onlineleader_targetstats_ens2_b07_w050.csv` |

The validation-aligned `b07` source is 70% target-statistics model and 30% established direct baseline. Raw-source 2.5%/5% probes are also available, but begin with the aligned seed77 and ensemble candidates because local proxy-to-LB transfer has historically been noisy.

## Primary Models

## New Dual-Head Candidates

The dual-head family trains separate direct-target Lossguide63 models for official observed and masked states. It is the first new proxy winner after the public LB calibration update.

| Family | Proxy evidence | Candidate files |
|---|---:|---|
| Dual-head seed2027 | **0.657815567** | `submission_dualhead_seed2027_{blend0.3,blend0.4,raw}.csv` |
| Dual-head seed77 | `0.658246481` | `submission_dualhead_seed77_{blend0.3,blend0.4,raw}.csv` |
| Dual-head seed42 | `0.658344567` | `submission_dualhead_seed42_{blend0.3,blend0.4,raw}.csv` |

These are candidate-only. No online submission has been made for this family.

## New Horizon-Head Candidates

The confirmed online best is now `submission_horizon_heads_seed2027_blend0.3` with LB RMSE `0.744238228`.

| Family | Proxy evidence | Candidate files |
|---|---:|---|
| Five-head age0/1/2/3-6/7+ seed77 | **0.653535452** | `submission_horizon_variant_age0_1_2_3_6_7p_seed77_{blend0.3,blend0.4,raw}.csv` |
| Five-head age0/1/2/3-6/7+ seed2027 | `0.653685113` | `submission_horizon_variant_age0_1_2_3_6_7p_seed2027_{blend0.3,blend0.4,raw}.csv` |

Five-seed averaged candidates, using seeds 77/2027/123/42/9001:

| Candidate | Pattern-CV guidance |
|---|---:|
| `submission_horizon5_ens5_blend0.3.csv` | Primary ensemble candidate |
| `submission_horizon5_ens5_blend0.4.csv` | More model weight, secondary |
| `submission_horizon5_ens5_plus_fourhead_blend0.7.csv` | Conservative cross-family candidate |

The horizon-head family routes direct-target models by `TWS_t_age`: observed, short gap, medium gap, and long gap. It is currently the strongest local proxy family.

| Family | Proxy evidence | Candidate files |
|---|---:|---|
| Horizon heads seed2027 | **0.653733980** | `submission_horizon_heads_seed2027_{blend0.3,blend0.4,raw}.csv` |
| Horizon heads seed77 | `0.653879803` | `submission_horizon_heads_seed77_{blend0.3,blend0.4,raw}.csv` |

These are candidate-only. No online submission has been made for this family.

| Family | Proxy evidence | Candidate files |
|---|---:|---|
| Corrected 1.5M, seed2027 | 0.658808476 (best) | `submission_cb90_d6_rows1500k_seed2027_{blend0.3,blend0.4,blend0.5,raw}.csv` |
| Corrected 2M, seed2027 | 0.659347945 | `submission_cb90_d6_rows2m_seed2027_*.csv` |
| Corrected 2M, seed77 | 0.659159112 | `submission_cb90_d6_rows2m_seed77_*.csv` |
| Corrected all-history mask90, seed77 | 0.659159112 | `submission_cb90_d6_rowsall_seed77_*.csv` |
| Corrected Huber, seed123 | 0.659881465 | `submission_cb90_d6_huber_rowsall_seed123_*.csv` |
| Corrected all-history Lossguide63, seed2027 | **0.658408041** | `submission_cb90_all_lossguide63_seed2027_*.csv` |
| Corrected all-history Lossguide63, seed77 | 0.658664605 | `submission_cb90_all_lossguide63_seed77_*.csv` |

## Window Blends

The `submission_window_*.csv` files mix only corrected 1.5M, 2M, and 1M families. `blend0.3` and `blend0.4` are the first weights to test because the fold-wise direct baseline blend selected 0.3/0.4 on the two tail periods.

## Current Shrinkage Candidates

Experiment #131 is the current local proxy leader. It applies fit-only per-age mean-centering and shrinkage to the five horizon heads, then blends each age bucket with the direct baseline.

Recommended manual order:

1. `submission_horizon_shrink_seed2027_calibrated.csv`
2. `submission_shrink_ens3.csv`
3. `submission_shrink_ens3_control25.csv`
4. `submission_shrink_ens5.csv`
5. `submission_shrink_ens2_horizon5_75.csv`
6. `submission_horizon_shrink_seed123_calibrated.csv`

Low-weight heterogeneity candidates:

- `submission_shrink_depthwise10.csv`
- `submission_shrink_depthwise20.csv`
- `submission_shrink_depthwise_seed77_10.csv`

Other-model low-weight candidates:

- `submission_shrink_ordered10.csv`
- `submission_shrink_ordered20.csv`
- `submission_shrink_lgbm10.csv`
- `submission_shrink_lgbm20.csv`

These are candidate-only and have not been submitted. All have 280,961 ordered IDs and finite target values. The calibrated files use only training-fold-derived age means and validation-selected fixed weights; no future TWS or test labels are used.

## Public-Pattern Long-Gap Calibration

Exact replay of the public 18-month mask chronology improved the long-gap bucket on the `2007-09` anchor for all three tested seeds and was neutral on `2009-01`. These are candidate-only, not online results:

- `submission_horizon_shrink_seed2027_gap7p_a075.csv`
- `submission_horizon_shrink_seed77_gap7p_a075.csv`
- `submission_horizon_shrink_seed123_gap7p_a075.csv`
- `submission_horizon_shrink_seed42_gap7p_a075.csv`
- `submission_horizon_shrink_seed9001_gap7p_a075.csv`

The corresponding `*_a0875.csv` files are a milder secondary calibration. All ten files have 280,961 ordered IDs and finite predictions. They differ from the calibrated controls only for the `gap7plus` bucket; no test labels or future TWS were used.

## New Heterogeneous Probe

CatBoost SymmetricTree/Plain is the first post-shrink route with a slightly better two-fold proxy than the shrink reference on seed77 (`0.652185260`). Remote full candidates are generated and being synced:

- `submission_horizon_plain_seed77_blend0.3.csv`
- `submission_horizon_plain_seed77_blend0.4.csv`
- `submission_horizon_plain_seed77_blend0.6.csv`
- `submission_horizon_plain_seed2027_blend0.3.csv`

Use these only as 5-10% blends with the online-best `submission_horizon_heads_seed2027_blend0.3.csv` until an online score confirms transfer. Do not submit `submission_horizon_shrink_seed2027_calibrated.csv` again; its observed LB was `0.749998587`.

Generated conservative control blends (all 280,961 rows, ordered IDs, finite predictions):

- `submission_onlinecontrol_plain_plain_seed77_w05.csv`
- `submission_onlinecontrol_plain_plain_seed77_w10.csv`
- `submission_onlinecontrol_plain_plain_ens2_w05.csv`
- `submission_onlinecontrol_plain_plain_ens2_w10.csv`

The `w15` and `w25` variants are exploratory only; start manual submissions with `w05`, then `w10`.

## Current Proxy Leader: Plain l2=5

The best current two-fold proxy is `SymmetricTree + Plain`, depth8, `l2_leaf_reg=5`, seed77 (`0.651949632`). Candidate-only control blends are validated with 280,961 ordered IDs and finite values:

- `submission_onlinecontrol_plain_l2_5_seed77_w05.csv`
- `submission_onlinecontrol_plain_l2_5_seed77_w10.csv`
- `submission_onlinecontrol_plain_l2_5_ens2_w05.csv`
- `submission_onlinecontrol_plain_l2_5_ens2_w10.csv`

These should be tested only after the online-best horizon control; the model itself has not been submitted online.

Extra l2=5 seed candidates are also synced for later ensemble construction:

- `submission_horizon_plain_l2_5_extra_seed123_blend0.3.csv`
- `submission_horizon_plain_l2_5_extra_seed9001_blend0.3.csv`

## Ordered l2=5 Heterogeneous Candidates

Ordered boosting with the same five age buckets and `l2_leaf_reg=5` gave the current best local proxy on seed77 (`0.651800227`; seed2027 `0.652853186`). These remain candidate-only and should be tested conservatively against the confirmed online control:

- `submission_onlinecontrol_ordered_l2_5_seed77_w05.csv`
- `submission_onlinecontrol_ordered_l2_5_seed77_w10.csv`
- `submission_onlinecontrol_ordered_l2_5_ens2_w05.csv`
- `submission_onlinecontrol_ordered_l2_5_ens2_w10.csv`

The underlying raw/blend0.3/0.4/0.6 files for both seeds are also retained in `submission/`; begin manual tests at 5% and 10% because local proxy-to-LB transfer is not yet calibrated.

## Public Reference Pipeline Candidates

The repository `external_refs/TWS-Drought-Forecast` contains a public same-challenge implementation and a prediction parquet with all 280,961 test IDs. It uses exact-month lags, SPEI/soil-moisture derived features and LightGBM. The imported file is not treated as a validated winner; it is staged only for diversity testing:

- `submission_onlinecontrol_external_toprepo_w05.csv`
- `submission_onlinecontrol_external_toprepo_w10.csv`
- `submission_external_toprepo_lgbm.csv` (raw external reference, diagnostic only)

The external 5% blend was submitted and scored `0.745795624`; treat this public-reference family as rejected for primary use.

## Enhanced Public-Feature LightGBM Candidates

The public reference features were rebuilt locally with exact-month TWS lags, lag rolling/momentum, SPEI divergence/severity/change and soil-moisture change/interaction. The seed2027 two-fold proxy is `0.651637084`, currently best locally. Start with conservative control blends:

- `submission_onlinecontrol_lgbm_enhanced_exact_seed2027_w05.csv`
- `submission_onlinecontrol_lgbm_enhanced_exact_seed2027_w10.csv`
- `submission_onlinecontrol_lgbm_enhanced_exact_ens2_w05.csv`
- `submission_onlinecontrol_lgbm_enhanced_exact_ens2_w10.csv`

## Enhanced LightGBM Variant Candidates

The leaves31/lr0.045/min_child60 variant improved the two-fold proxy to seed9001 `0.651391903` (seed42 `0.651480013`). Candidate files are synced and should be tested only as low-weight additions to the current online leader:

- `submission_onlineleader_lgbm_variant_seed9001_w025.csv`
- `submission_onlineleader_lgbm_variant_seed9001_w050.csv`
- `submission_onlineleader_lgbm_variant_seed42_w025.csv`
- `submission_onlineleader_lgbm_variant_seed42_w050.csv`

Use 2.5% first, then 5%; 7.5-15% are exploratory only. Do not submit the raw variant first.

## Enhanced CatBoost Candidates

The corrected exact-feature CatBoost Ordered run is currently the strongest local proxy: seed2027 `0.650565407`, seed77 `0.650615096`. Test low-weight blends against the online leader in this order:

- `submission_onlineleader_catboost_enhanced_seed2027_w025.csv`
- `submission_onlineleader_catboost_enhanced_seed2027_w050.csv`
- `submission_onlineleader_catboost_enhanced_seed77_w025.csv`
- `submission_onlineleader_catboost_enhanced_seed77_w050.csv`

All ten CatBoost control blends are row/ID/finite-value validated; do not submit raw CatBoost first.

## Enhanced XGBoost Diversity Candidates

Enhanced XGBoost is weaker locally (seed9001 `0.652466615`) but is retained for optional low-weight diversity only:

- `submission_onlineleader_xgb_enhanced_seed9001_w025.csv`
- `submission_onlineleader_xgb_enhanced_seed9001_w050.csv`
- `submission_onlineleader_xgb_enhanced_seed42_w025.csv`

Do not test XGBoost before the stronger CatBoost blends.

## Enhanced CatBoost Depth6 Candidates

Depth6 Plain is the strongest secondary local partner so far (seed2027 `0.650860793`, seed77 `0.651324434`). Test only after Ordered blends:

- `submission_onlineleader_catboost_depth6_seed2027_w025.csv`
- `submission_onlineleader_catboost_depth6_seed2027_w050.csv`
- `submission_onlineleader_catboost_depth6_seed77_w025.csv`

These eight control blends are row/ID/finite-value validated; raw depth6 files are diagnostic only.

## Enhanced LightGBM GOSS Diversity Candidates

The corrected GOSS wrapper completed both proxy seeds but remained behind Ordered (seed77 `0.651641953`, seed2027 `0.652527468`). Keep these only as optional 2.5-5% diversity probes against the confirmed online leader; do not submit raw GOSS files first:

- `submission_onlineleader_lgbm_goss_seed77_w025.csv`
- `submission_onlineleader_lgbm_goss_seed77_w050.csv`
- `submission_onlineleader_lgbm_goss_seed2027_w025.csv`
- `submission_onlineleader_lgbm_goss_seed2027_w050.csv`

All GOSS source and online-leader blends have been validated for 280,961 rows, exact SampleSubmission ID order, and finite predictions.

Ordered extra seeds123/9001 were also completed (`0.652054370`/`0.651687405`) and are available only as optional low-weight blends:

- `submission_onlineleader_catboost_ordered_extra_seed123_w025.csv`
- `submission_onlineleader_catboost_ordered_extra_seed9001_w025.csv`
- `submission_onlineleader_catboost_ordered_extra_seed123_w050.csv`
- `submission_onlineleader_catboost_ordered_extra_seed9001_w050.csv`

Bayesian bootstrap CatBoost seed321 was weaker locally (`0.652389367`) but has been staged only for optional 2.5-5% diversity:

- `submission_onlineleader_catboost_bayes_seed321_w025.csv`
- `submission_onlineleader_catboost_bayes_seed321_w050.csv`

Seed777 full candidates are now synchronized and its optional blends are available:

- `submission_onlineleader_catboost_bayes_seed777_w025.csv`
- `submission_onlineleader_catboost_bayes_seed777_w050.csv`

Low-temperature Bayesian Ordered candidates are also staged as optional diversity:

- `submission_onlineleader_catboost_bayes_lowtemp_seed42_w025.csv`
- `submission_onlineleader_catboost_bayes_lowtemp_seed42_w050.csv`
- `submission_onlineleader_catboost_bayes_lowtemp_seed9001_w025.csv`
- `submission_onlineleader_catboost_bayes_lowtemp_seed9001_w050.csv`

Plain Bayesian candidates are also available as optional low-weight diversity:

- `submission_onlineleader_catboost_plain_bayes_seed123_w025.csv`
- `submission_onlineleader_catboost_plain_bayes_seed123_w050.csv`
- `submission_onlineleader_catboost_plain_bayes_seed2027_w025.csv`
- `submission_onlineleader_catboost_plain_bayes_seed2027_w050.csv`

Extra-seed XGBoost RMSE blends are available only for optional diversity:

- `submission_onlineleader_xgb_extra_seed27182_w025.csv`
- `submission_onlineleader_xgb_extra_seed27182_w050.csv`
- `submission_onlineleader_xgb_extra_seed31415_w025.csv`
- `submission_onlineleader_xgb_extra_seed31415_w050.csv`

Ordered Bernoulli CatBoost blends are also staged for optional low-weight testing:

- `submission_onlineleader_catboost_bernoulli_seed9001_w025.csv`
- `submission_onlineleader_catboost_bernoulli_seed9001_w050.csv`
- `submission_onlineleader_catboost_bernoulli_seed42_w025.csv`
- `submission_onlineleader_catboost_bernoulli_seed42_w050.csv`

## Enhanced XGBoost Depth8 Diversity Candidates

Depth8 did not improve the enhanced XGBoost proxy (seed42 `0.652626130`, seed9001 `0.652635715`, versus depth6 best `0.652466615`). Retain only these low-weight diagnostic blends; they are not ahead of the CatBoost primary queue:

- `submission_onlineleader_xgb_depth8_seed42_w025.csv`
- `submission_onlineleader_xgb_depth8_seed42_w050.csv`
- `submission_onlineleader_xgb_depth8_seed9001_w025.csv`
- `submission_onlineleader_xgb_depth8_seed9001_w050.csv`

All 8 source files and 6 low-weight blends have 280,961 rows, exact SampleSubmission ID order, and finite predictions.

## Exclusion Rule

Do not use archived provisional `rows1500k/rows1750k/rows1m/rows2m` files from before the `window-aligned-v2` final-fit correction. A direct old-versus-new comparison had prediction-delta RMSE `0.021667641`, so those files are not interchangeable with the corrected candidates.

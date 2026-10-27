## Latest decision — 2026-09-10

Stop current-stack center/scale experiments after #256/#257 degraded both seeds. Prioritize a faithful sparse-calendar recursive validation harness and a CV gate before generating test predictions. Neither new candidate has an online score.

#258 implements that harness and gate and is RUNNING from 13:38 CST as `water_gpu_sparse_replay_control` (three anchors, seeds2027/77, 500 fixed iterations). Next: measure a time-aligned feedback variant against its complete reference, keeping all other settings fixed. Current baseline intentionally preserves the legacy forecast(t+1)-into-row-t heuristic. Historical RUNNING labels below are not live process status.

## High Priority

1. **[Model / C] GPU CatBoost**: depth 6/8/10 with early stopping on two deterministic tail folds; blend only from validation predictions.
2. **[Model / C] RUNNING** GPU XGBoost: mask90/all-history depth 6/8/10 histogram models with regularization and two-fold stability checks under `water_gpu_xgb_all`.
3. **[Ensemble / C] Cross-family blend**: combine LB-best ExtraTrees/Ridge predictions with the most stable GPU model; save several nearby weights for manual submission.

4. **[Capacity / C] RUNNING**: Lossguide `max_leaves=95/127` and lr0.06/l2=5 cross against the proxy-best Lossguide63 family.

5. **[Target / B] COMPLETED/PROMISING**: masked/observed dual-head direct-target model; seed2027 proxy `0.657816`, full candidates synced. Next validate horizon-specific routing before online submission.

## Medium Priority

6. **[Mask / B] CatBoost mask-rate sweep**: compare 0.35/0.50/0.665 augmentation only after selecting a GPU model family.
5. **[Loss / D] GPU robust objective**: CatBoost Huber or XGBoost pseudo-Huber after direct RMSE models establish a reference.

## New Sweep Queue

16. **[Horizon / B] COMPLETED/PROMISING**: five direct heads split age1 and age2 (`age0/1/2/3-6/7+`) under the official month-level mask schedule; seed77 proxy `0.653535452`.
17. **[Ensemble / C] RUNNING**: train extra five-head seeds 123/42/9001 and compare seed averaging against the confirmed four-head LB winner.
18. **[Validation / B] QUEUED**: replay the exact 18-month public chronology and mask states on historical windows (`2007-09`/`2009-01` anchors); use this to select blend weights and reject locally strong but LB-weak models.
19. **[Training Window / A] RUNNING**: five-head final fit restricted to the latest 500k training rows, aligned with pattern-CV fit window; compare against all-history ensemble.
20. **[Training Window / A] RUNNING**: five-head final fit restricted to the latest 1M training rows as a midpoint between all-history and recency500.
21. **[Model / C] RUNNING**: XGBoost GPU direct-target five-head router under the official mask schedule; evaluate as a heterogeneous blending partner rather than a direct CatBoost replacement.
22. **[Model / C] RUNNING**: deeper XGBoost horizon heads (`depth10/lr0.022/l2=30`) to test capacity beyond the first XGB configuration.
23. **[Horizon Capacity / C] DRY**: five age-routed CatBoost heads with Lossguide `max_leaves=95/127`; gains were mixed and below `0.0005`, retain candidates only for diversity.
24. **[Loss / D] REJECTED**: age-routed CatBoost Huber `delta=1` was materially worse than RMSE; retain only loss-diversity files.
25. **[Post-process / D] COMPLETED/PROMISING**: age-conditioned mean-centering/shrinkage produced proxy `0.652463794`; validate against public-pattern chronology and blend with #117 control.
26. **[Validation / B] QUEUED**: score #129 calibrated candidates with equal-fold and public-pattern metrics before any manual LB recommendation; preserve #117 as control.
27. **[Model / C] COMPLETED**: CatBoost age-routed heads with `boosting_type=Ordered`; stable but weaker than shrink, retained for low-weight diversity.
28. **[Model / C] COMPLETED**: LightGBM GPU age-routed heads; close to CatBoost horizon but weaker than shrink, retained for low-weight diversity.
29. **[Post-process / D] QUEUED**: LightGBM/Ordered residual corrector only after manual LB feedback identifies whether shrinkage over/under-corrects.
30. **[Validation / B] COMPLETED/PROMISING**: exact public-pattern replay supports long-gap `alpha=0.75`; candidate files are in `submission/` and must be tested only after the current shrink control.
31. **[Model / C] RUNNING**: CatBoost SymmetricTree with `boosting_type=Plain` across the five age heads; use as a low-weight heterogeneous partner if its public/tail proxy is complementary.
32. **[Post-process / D] QUEUED**: combine the best long-gap calibrated candidate with Ordered/LightGBM at 5-10% weights after manual LB feedback.
33. **[Online Calibration / B] REJECTED**: ordinary age-conditioned shrinkage scored `0.749998587` online despite a strong local proxy; do not promote shrink-only submissions without a control blend.
34. **[Model / C] PROMISING/RUNNING**: SymmetricTree/Plain seed77 is the best current heterogeneous proxy; finish extra seeds 123/9001 and build conservative control blends.
35. **[Ensemble / C] RUNNING**: Plain seed77/2027 5-25% blends against the online-best horizon control are generated and validated; only low weights are eligible before LB evidence.
36. **[Capacity / C] RUNNING**: SymmetricTree/Plain depth6 cross; promote only if its two-fold mean beats depth8 Plain or supplies clear diversity for a 5% blend.
37. **[Regularization / C] REJECTED**: Plain Bernoulli subsample0.8 reached `0.653523510`; do not repeat bootstrap subsampling on this horizon family.
38. **[Hyperparameter / C] RUNNING**: Plain depth8 l2=5 cross; stop if both seeds fail to beat `0.652185260` by the promotion threshold.
39. **[Ensemble / C] PROMISING**: Plain l2=5 seed77/2027 control blends generated at 5/10/15/20/30%; manual order starts at 5% and 10% because local-vs-LB calibration is still uncertain.
40. **[Model / C] COMPLETED/PROMISING**: Ordered CatBoost l2=5 horizon heads; seed77 proxy `0.651800227`, seed2027 `0.652853186`; eight candidates plus conservative control blends synced.
41. **[Model / C] RUNNING**: XGBoost CUDA horizon heads, depth6/lr0.045/max_bin512 with lambda12/alpha0.1 and seeds42/9001; use as a cross-family probe and low-weight blend source.
42. **[Feature / A] QUEUED FROM PUBLIC SOLUTION**: Rebuild exact-month TWS lags (1/2/3), lag momentum/rolling stats, SPEI divergence/severity, SPEI/soil-moisture changes and same-time spatial aggregates; evaluate with the official mask schedule before any full fit.
43. **[Ensemble / C] READY**: Public same-challenge LightGBM predictions imported; test 5%/10% blends first because external target variance is materially lower than the CatBoost horizon family.
44. **[Feature + Model / A-C] COMPLETED/PROMISING**: GPU LightGBM five-head router with exact-month lags, drought divergence/severity and soil-moisture interaction features; seed2027 proxy `0.651637084`, new local best; 8 candidates and 12 blends synced.
45. **[Model / C] RUNNING NEXT**: PyTorch MLP horizon model with bucket one-hot, LayerNorm/GELU, SmoothL1 loss and official mask schedule; retain as an ensemble partner if proxy is competitive or structurally complementary.
46. **[Online / C] CONFIRMED**: Ordered l2=5 seed77 at 5% control is the current LB anchor (`0.744166455`); calibrate all future model additions against this anchor, not the older raw horizon control.
47. **[Online / C] REJECTED**: External same-challenge repository prediction at 5% scored `0.745795624`; stop recommending external-toprepo blends as primary.
48. **[Model / C] REJECTED**: PyTorch MLP horizon probe reached proxy `0.663519`; retain only as a low-weight diversity artifact, do not spend more GPU cycles on this architecture.
49. **[Model / C] RUNNING**: Enhanced LightGBM leaves31/lr0.045/min_child60 variant with exact public features; promote only if it beats `0.651637084` or supplies useful online-calibrated diversity.
50. **[Model / C] PROMISING**: Enhanced LightGBM variant beat the prior proxy (`0.651391903` vs `0.651637084`); test only conservative 2.5-15% blends against the online leader before any promotion.
51. **[Model / C] RUNNING**: Enhanced CatBoost Ordered/SymmetricTree heads on the same exact public features; compare with LightGBM for cross-family diversity.
52. **[Model / C] PROMISING**: Enhanced CatBoost correctly wired to exact public features; seed2027 proxy `0.650565407` is the current local best. Test 2.5% blend first, then 5% if LB confirms transfer.
53. **[Model / C] RUNNING**: Enhanced XGBoost CUDA heads on exact public features; retain only if it beats `0.650565407` or offers useful residual diversity.
54. **[Model / C] REJECTED/PARTNER**: Enhanced XGBoost reached seed9001 `0.652466615`; do not promote as primary, but keep 2.5-7.5% blends for optional diversity.
55. **[Model / C] RUNNING**: Enhanced CatBoost Lossguide (`max_leaves=63`, l2=10, lr0.035) on exact features; this tests growth-policy diversity after Ordered's local win.
56. **[Model / C] REJECTED**: Enhanced Lossguide seeds77/2027 reached `0.652123640`/`0.652352165`; stop Lossguide promotion and retain no primary submission.
57. **[Model / C] RUNNING**: Enhanced CatBoost depth6 Plain/SymmetricTree (`lr0.05`, l2=5) tests lower-capacity heads after depth8 Ordered won.
58. **[Model / C] COMPLETED/PARTNER**: Enhanced depth6 Plain seed2027 `0.650860793` is close but below Ordered; retain 2.5-10% blends for optional diversity, do not replace the anchor.
59. **[Model / C] REJECTED**: Enhanced LightGBM GOSS proxy seed77 `0.651641953` and seed2027 `0.652527468` did not beat Ordered; keep only optional low-weight files if generated.
60. **[Engineering / C] COMPLETED**: LightGBM 4.x compatibility shim translates legacy early stopping and best-iteration APIs; GOSS pair proxy and full candidates completed.
61. **[Regularization / C] QUEUED**: Enhanced CatBoost Ordered with Bayesian bootstrap `bagging_temperature=2.0`, seeds321/777, starts automatically after Ordered extra seeds.
62. **[Model / C] REJECTED/PARTNER**: Enhanced Ordered extra seeds123/9001 reached `0.652054370`/`0.651687405`; retain only low-weight diversity files and do not promote over seeds77/2027.
63. **[Regularization / C] REJECTED/PARTNER**: Bayesian bootstrap temperature2.0 seed321 `0.652389367`, seed777 `0.651038728`; keep seed777 only for optional low-weight diversity after full fit.
64. **[Model / C] QUEUED**: LightGBM DART exact-feature five-head model, seeds31415/27182, starts automatically after Bayesian CatBoost.
65. **[Regularization / C] QUEUED**: GPU-native CatBoost Ordered Bayesian `bagging_temperature=0.5`, seeds42/9001, starts automatically after DART to keep the RTX 4090 occupied.
66. **[Model / C] REJECTED**: LightGBM DART diagnostic fold blend `0.725326086`; CPU-only/no early-stopping behavior on this build, stop further DART runs.
67. **[Model / C] REJECTED/PARTNER**: Plain/SymmetricTree CatBoost with Bayesian temperature0.5, seed123 `0.651577409`, seed2027 `0.651282548`; retain only low-weight diversity files.
68. **[Model / C] RUNNING**: Ordered/SymmetricTree CatBoost with MVS bootstrap `subsample=0.8`, seeds42/9001, started automatically after Plain Bayesian.
69. **[Model / C] REJECTED**: Ordered MVS `subsample=0.8` and `0.95` both produced fold0 blend around `0.7257`; abandon this sampling route.
70. **[Loss / D] RUNNING**: GPU XGBoost pseudo-Huber (`reg:pseudohubererror`) exact-feature horizon heads, seeds31415/27182.
71. **[Loss / D] REJECTED**: Pseudo-Huber seed31415 proxy `0.653038051`, materially behind RMSE XGBoost; stop robust-loss route.
72. **[Model / C] REJECTED/PARTNER**: GPU XGBoost squarederror extra seeds31415 `0.652884192` and 27182 `0.652035512`; keep only low-weight diversity files.
73. **[Model / C] REJECTED/PARTNER**: Ordered/SymmetricTree CatBoost Bernoulli seed42 `0.651899838`, seed9001 `0.651441649`; retain only low-weight diversity files.
74. **[Model / C] COMPLETED/REJECTED**: Enhanced XGBoost depth8 CUDA heads scored seed42 `0.652626130` and seed9001 `0.652635715`, both behind depth6; sync completed candidates but do not refine depth.
75. **[Regularization / C] COMPLETED/REJECTED**: Enhanced Ordered CatBoost `random_strength=0.25` scored seed77 `0.651701356`, seed2027 `0.651256048`; both trail the control, retain only low-weight diversity.
76. **[Regularization / C] COMPLETED/DRY**: Enhanced Ordered CatBoost `random_strength=2.0` scored seed77 `0.651342272`, seed2027 `0.651285425`; together with failed 0.25 endpoint, stop random-strength tuning.
77. **[Capacity / C] COMPLETED/DRY**: Enhanced XGBoost depth5 scored seed27182 `0.652285351`, seed2027 `0.652676446`; with failed depth8, stop XGBoost capacity tuning around depth6.
78. **[Target Statistics / A+B] QUEUED**: M5-style past-only coordinate and coordinate-month target mean/std/count in the strongest Enhanced Ordered heads; training uses expanding shift and validation/test map only fit-period labels.
79. **[Regional Model / A+C] COMPLETED/PARTNER**: 6 spherical regions crossed with TWS-age heads reached `0.650136258`, a sub-threshold `0.000429149` gain over ordinary Enhanced Ordered but far behind block-frozen target history; keep diversity and test only one coarse-region + block-history combination.
80. **[Sequence Model / B+C] COMPLETED/DIVERSITY**: 12-month Temporal CNN reached `0.653780593`; retain only conservative online-anchor blends after the missing-anchor full retry, then evaluate local-normalization and graph fixes.
81. **[Categorical Entity / A+C] COMPLETED/DIVERSITY/DRY**: fixed coordinate and coarse-band Ordered CTRs reached seed77 `0.650168492`, seed2027 `0.651014623`; both trail the ordinary/blocked-history leaders, so keep optional `w025` diversity files but stop CTR-only refinement.
82. **[Entity Ensemble / A+B+C] COMPLETED/DRY**: categorical CTR + block-frozen target history reached seed2027 `0.641651029`, seed77 `0.642175262`; both trail pure block-frozen, so stop CTR complexity/band refinements.
83. **[Graph Sequence / A+B+C] QUEUED**: fixed spherical 5-NN aggregation of same-month TWS/SPEI/soil followed by 12-month Temporal CNN; no target or future-time graph messages.
84. **[Target Statistics / A+B] FAILED**: last-target plus 12/24 windows gave fold0 `0.721859198`; removing last-target still scored `0.723445127`, so the naive recent-window encoding is DRY due train/query availability mismatch.
85. **[Target Statistics / A+B] COMPLETED/NEW PROXY LEADER**: freeze last/rolling12/rolling24 statistics at each 12-month pseudo-forecast block start; seed77 `0.641194192`, seed2027 `0.639744590`, improving both base target-stat seeds by `0.0084-0.0106`; stage conservative aligned candidates.
85b. **[Online / A+B] CONFIRMED**: seed2027 `b08` weight ramp is monotonic through 2.5/5/7.5/10%. 10% scored LB **`0.741526`**, beating Ordered l2=5 5% by `0.002640455`. Next isolate 15% (`..._w150.csv`), then 20% if it still gains. Do not mix seed ensembles or XGB/LGBM until the weight curve flattens.
86. **[Target Formulation / B] COMPLETED/BORDERLINE**: fixed-18 failed; dynamic span ordinary CV reached `0.641422203`, but exact 18-date/40-month replay gave block40 an overall gain `0.000499874` and masked gain `0.000550866`. Keep only seed2027 `b08_w025` behind block12 candidates; stop block-length tuning.
87. **[Target Formulation / B+D] HIGH PRIORITY**: Web-Traffic-style entity normalization: predict `(target - frozen_coordinate_center) / frozen_coordinate_scale`, then reconstruct with fit-only block-frozen statistics; use robust scale clipping and no query labels.
88. **[Sequence / B+C] QUEUED**: Web-Traffic/Favorita-style mask-pattern-aligned sequence-to-sequence model with coordinate normalization and shared 18-horizon outputs; compare median seed ensemble against the existing one-step TCN.
89. **[Model / A+B+C] QUEUED**: LightGBM leaves31 exact-feature heads with strict block-frozen target memory, seed9001; full-fit supervised augmentation hook is regression-tested and the service is wired after localnorm TCN.
90. **[Graph Sequence / A+B+C] COMPLETED/PROMISING DIVERSITY**: fixed spherical 5-NN TCN reached `0.648696594`, improving original TCN by `0.005083999`; 4 files validated, keep `onlineleader_w05` behind block-frozen primary submissions.
91. **[Target Formulation / B+D] REPAIR QUEUED**: naive center/scale normalization failed at `0.656434316` because early pseudo-block std is zero; run identity fallback for `std<0.10`, then center-only residuals as two minimal root-cause fixes.
92. **[Regional Model / A+C] COMPLETED/DRY**: block-frozen region3 reached `0.641061282`; mixed-fold diversity but worse overall, and region6 was also suboptimal. Keep only `blockregion3...w025`, stop region-count tuning.
93. **[Model / A+B+C] COMPLETED/PROMISING DIVERSITY**: block-memory XGBoost reached `0.642175137`, a `0.010291478` gain over ordinary enhanced XGBoost; retain validated `onlineleader...w025` as cross-family partner.
94. **[Sequence / B+C] COMPLETED/REJECTED**: sample-wise local-normalized TCN reached `0.656093251`; mixed folds and worse pair than original TCN. Stop this route, retain graph TCN as the successful sequence repair.
95. **[Model / A+B+C] COMPLETED/PROMISING DIVERSITY**: block-memory LightGBM reached `0.642518992`, improving its ordinary variant by `0.008872911`; validated `b075_w025` is a third-family probe after CatBoost/XGBoost.
96. **[Ensemble / C] COMPLETED/STABLE**: extra block-frozen CatBoost seeds123 `0.640458422`, seed9001 `0.641299445`; four-seed and best-three `b08_w025/w050` ensembles generated and validated.
97. **[Spatial / A] COMPLETED/NEW PROXY LEADER**: IDW/250km/neighbour SPEI-soil on the #213 Ordered family reached seed77 **0.629384** and seed2027 `0.630983`, beating #213 by 0.005349. GPU job `water_gpu_catboost_blockfrozen_nb_idw`.
98. **[Target / B] COMPLETED/REJECTED**: residual-around-neighbour-persist reached seed77 **0.647321** / seed2027 `0.648223`, worse than direct #213/#215 by ~0.018. Stop this target axis. GPU job `water_gpu_catboost_blockfrozen_nb_residual`.
99. **[Model / C] COMPLETED/DIVERSITY**: XGBoost CUDA + #213 neighbourhood reached seed2027 **0.638228** / seed27182 `0.638529`, beating old XGB block-frozen 0.642175 but trailing CatBoost #213/#215. GPU job `water_gpu_xgb_blockfrozen_neighborhood`.
100. **[Ensemble / C] COMPLETED/NOT PROMOTED**: extra #213 neighbourhood seeds123 `0.636391`, seed9001 `0.635754`; both trail seed2027 0.634733. Keep as four-seed material only. GPU job `water_gpu_catboost_blockfrozen_neighborhood_extra`.
101. **[Feature / A] COMPLETED/NEW PROXY LEADER**: ERA5 monthly on the #215 IDW Ordered family reached seed2027 **0.622694**. Online 30% scored **0.7311**. Further `w` ramps of this family cannot reach 0.6x.
102. **[Inference / A] COMPLETED**: recursive own-predictions into neighbourhood/TWS-lag features at test time, plus fit-only ERA5 calendar-month anomalies. Training proxy seed77 **0.614575** / seed2027 **0.614801**. Online 10% scored **0.7232**. Job `water_gpu_catboost_idw_era5_recursive` stopped.
103. **[Inference / A] COMPLETED/NEW ONLINE LEADER**: intra-month spatial iteration (3 inner predict/refresh cycles) plus train-time neighbour-persist fill. Training proxy seed77 **0.613860** / seed2027 **0.614622**. Online seed77 raw 10% scored **0.7179**. Job `water_gpu_catboost_idw_era5_spatialiter` stopped. Do not spend more `w` ramps.
104. **[Training Window / A] COMPLETED/NEW ONLINE LEADER**: pattern-mask validation + recency full-fit (Train time>=2010-01). Online seed77 raw 10% scored **0.7173**. Weak transfer (−0.0006 vs spatialiter 0.7179). Do not spend more pattern `w`. Job sleeper stopped.
105. **[Training Window / A] COMPLETED/LOW PRIORITY**: exponential year weights (half-life 3 years) on all-history full-fit. Test vs pattern RMS **0.159 / 0.161**. After 2012+ hurt, this is only an optional later probe, not the path to 0.6x. Job `water_gpu_catboost_idw_era5_weighted` stopped.
106. **[Inference / A] COMPLETED/DRY**: 5 intra-month spatial iterations on the all-history #232 family (was 3). Ordinary two-fold seed77 **0.613860** / seed2027 **0.614622**. Test vs 3-iter RMS **0.0166 / 0.0165** (corr 0.9997). Stop inner-iter count. Job `water_gpu_catboost_idw_era5_spatialiter5` stopped.
107. **[Training Window / A] COMPLETED/REJECTED/DRY**: recency full-fit Train time>=2012-01. Online seed77 raw 10% scored **0.7192**, worse than 2010+ 0.7173. Stop tightening the cutoff. Do not submit 2.5/5%. Job `water_gpu_catboost_idw_era5_recency2012` stopped.
108. **[Inference / A] COMPLETED/REJECTED**: train-time neighbour teacher two-pass on 2010+ recency. Test std **1.63** vs spatialiter **0.70**, RMS **1.589**, corr **0.36**. Pass-2 saw teacher neighbourhoods at train but last-obs neighbourhoods at test first-iter. Stop this two-pass design. Job `water_gpu_catboost_idw_era5_teacher2pass` stopped.
109. **[Model / C] COMPLETED/NEW ONLINE LEADER**: XGBoost CUDA on the IDW+ERA5+spatialiter+2010 stack. Online seed77 raw 10% scored **0.715**. Do not spend more XGB `w`. Job sleeper stopped.
110. **[Training Window / A] COMPLETED/DRY**: 2010+ recency only on gap2/gap3_6. Sequential test preds identical to spatialiter (RMS 0). Static differed (RMS 0.11); the recursive walk washed it out. Job stopped.
111. **[Ensemble / C] COMPLETED/REJECTED**: extra 2010+ CatBoost ens4 2.5% scored **0.7153** online, worse than 0.715. Do not try 5/10%.
112. **[Ensemble / C] COMPLETED/NEW ONLINE LEADER**: extra XGB ens2 2.5% scored **0.7147**. Do not spend more extra-XGB `w`.
113. **[Model / C] COMPLETED/NEW ONLINE LEADER**: LightGBM on the IDW+ERA5+spatialiter+2010 stack. Online seed77 2.5% scored **0.7143**, seed2027 2.5% scored **0.7139**. Do not spend more LGBM `w`. Job stopped.
114. **[Ensemble / C] COMPLETED**: extra LGBM ens2 2.5% scored **0.7140**, trailing seed2027 0.7139. Do not spend more extra-LGBM `w`. Job stopped.
115. **[Feature / A] COMPLETED/REJECTED**: lagged ERA5 calendar-month anomalies on CatBoost 2010+. Local ordinary **0.61049 / 0.60954** beat #233, but online seed2027 2.5% scored **0.7142**. Do not try 5/10% or seed77 `w`. Local two-fold gain did not transfer.
116. **[Model / C] STOPPED/INCOMPLETE**: XGB lag-anom seed2027 CSVs local; pattern worse than #241. seed77 died at 2017-06. Do not restart.
117. **[Inference / A] COMPLETED/TIE**: mix existing 2010+ **static** (no recursive walk) into 0.7139. Online seed77 2.5% scored **0.7139**. Recursion-off at 2.5% did not move public. Do not spend static 5/10%.
118. **[Inference / A] COMPLETED/#262**: time-aligned recursive on #258 heads. Pooled aligned **0.522228** vs static **0.526434** vs recursive **0.542328**; masked **0.548073** vs **0.554980**. Gate passed. 4/6 beat static; 2009-01 both seeds lost ~+0.012.
119. **[Inference / A] COMPLETED/NEW ONLINE LEADER**: 2010+ CatBoost aligned Test walk. Online seed77 2.5% scored **0.7121**, seed2027 2.5% scored **0.7105**. Do not spend more CatBoost aligned `w`.
120. **[Inference / A] COMPLETED/NEW ONLINE LEADER**: LGBM 2010+ aligned Test walk. Online seed2027 2.5% scored **0.7090**, seed77 2.5% scored **0.7076**. Do not spend LGBM aligned 5/10%.
121. **[Search / A] STOPPED**: Extra CatBoost/LGBM aligned COMPLETE (same family). Extra LGBM seed123 2.5% tied 0.6955. Extra XGB STOPPED incomplete (~4/18). 2 slots. Do not submit extra CatBoost/LGBM 9001.
122. **[Online / A] COMPLETED/#273**: XGB aligned seed77 w 10/20/30 then +20% into 0.6957 scored **0.7027 / 0.6970 / 0.6957 / 0.6955**. Saturated. Do not submit w500.
123. **[Online / A] REJECTED/#275**: XGB aligned seed2027 2.5% into 0.6955 scored **LB 0.6956**. No transfer. Do not spend seed2027 5/10%.
124. **[Online / A] REJECTED/#276**: neighbourhood persist a3 w15 on 0.6955 scored **LB 0.6965**. Do not spend persist w25.
125. **[Online / A] TIE/#277**: extra LGBM aligned seed123 2.5% scored **LB 0.6955**. No transfer. Do not spend extra LGBM 9001.

6. **[Hyperparameter / C]** Compare CatBoost d6 learning rate 0.06/l2=5 and l2=30 variants.
7. **[Hyperparameter / C] COMPLETED** CatBoost depth 7 and random_strength=0 did not beat `0.662865975`; do not repeat this exact sweep.
8. **[Mask / B] DRY** Mask35 failed; mask80 and mask90 set small new bests (`0.662714549`, then `0.662530442`) but both gains were `<0.0005`; do not refine random mask rates further.
9. **[Spatial / A] DRY** Spatial d6 and lr06 crosses set small new bests (`0.662121227`, then `0.661984201`) but gains stayed `<0.0005`; finish the already-started mask90 cross, then stop pure spatial tuning.

10. **[Missingness / B] DRY** Calendar default/lr06 failed; calendar+spatial reached `0.661820623` but improved by only `0.000163578`; do not continue calendar micro-tuning.

11. **[Training Window / A] RUNNING** All-history mask90 is now the clear winner (`0.659159112`, all five below `0.6597`); exploit lr/l2/depth variants and then ensemble stable seeds.

12. **[Missingness x Window / B] RUNNING** Compare 1.5M window and all-history mask80/95/100/calendar-spatial after all-history hyperparameter micro-tuning became DRY.

13. **[Regularization / C] QUEUED** On the winning all-history mask90 regime, compare CatBoost bagging temperature, random strength, leaf estimation iterations, Bernoulli subsampling, and Lossguide growth. This is a new model regularization dimension and is wired to start automatically after the active cross queue.

14. **[Audit / A] IN PROGRESS** Rebuild promoted window candidates with final-fit windows aligned to validation (`window-aligned-v2`); rows1500k, rows1750k, and rows1m are corrected and locally validated, while rows2m remains provisional until its new CV/promotion completes.
15. **[Horizon / B] COMPLETED/PROMISING**: age-bucket direct heads reached proxy `0.653734` (seed2027) and `0.653880` (seed77); candidates synced. Next use online LB calibration, then test finer age boundaries only if needed.

## Dry Decisions

- **L2 regularization sweep**: DRY after l2=20 and l2=30 multi-seed runs failed to beat `0.662865975`; do not repeat these settings.
- **Randomness/depth sweep**: 2/3 non-improving experiments after `random_strength=0` and depth7; one distinct nearby hyperparameter result remains before declaring the dimension DRY.
- **Random mask-rate sweep**: DRY after mask35 failed and mask80/mask90 improved by less than `0.0005`; do not test adjacent random mask rates.
- **Pure spatial sweep**: DRY after d6, d7, and lr06/l2=5 spatial variants all improved by less than `0.0005`; do not continue depth/lr micro-tuning without a new missingness mechanism.
- **Calendar-mask sweep**: DRY after default, lr06/l2=5, and spatial variants failed to improve by `0.0005`; keep the best diversity candidates but stop calendar micro-tuning.
- **All-history lr/depth sweep**: DRY after lr06, lr08, d7, and d8 moved the proxy by less than `0.0005`; do not add adjacent tree hyperparameters.
- **Enhanced Ordered random-strength sweep**: DRY after both `0.25` and `2.0` failed across seeds77/2027 versus the `1.0` control; retain only low-weight diversity and pivot to capacity/entity/sequence dimensions.
- **Enhanced XGBoost capacity sweep**: DRY after depth5/high-child and depth8 both failed to beat the depth6 reference; retain low-weight diversity only and pivot to entity/sequence models.
- **CatBoost lagged ERA5 calendar anomalies**: DRY for online mixing after seed2027 2.5% scored 0.7142 versus 0.7139. Local ordinary improved (~0.610 vs 0.614) but did not transfer. Do not try 5/10% or extra CatBoost seeds of this feature.

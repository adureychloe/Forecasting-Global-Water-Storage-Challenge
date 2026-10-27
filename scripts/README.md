# 脚本导航

从仓库根目录执行 `python scripts/<name>.py`。代码保留同级导入，数据、缓存和输出定位到仓库根目录。

## 最终方案：17 个模块

| 文件 | 用途 |
| --- | --- |
| `download_era5_monthly.py` | ERA5 下载；解包说明见 [TRAIN.md](../review/zindi_code_review/TRAIN.md) |
| `gpu_catboost_idw_era5_aligned.py` | CatBoost aligned 训练与推断 |
| `gpu_lgbm_idw_era5_aligned.py` | LightGBM aligned 训练与推断 |
| `gpu_lgbm_idw_era5_aligned_extra.py` | LightGBM extra seeds，最终采用 seed 123 |
| `gpu_xgb_idw_era5_aligned.py` | XGBoost aligned 训练与推断 |
| `aligned_feedback.py` | 按日历对齐的逐月反馈推断 |
| `coordinate_compliance.py` | 最终模型的坐标特征移除逻辑 |
| `baseline.py`、`enhanced_features.py` | 基础数据工具与时间特征；baseline 可用 `--help` 查看参数 |
| `era5_cell_features.py`、`era5_anomaly_features.py` | ERA5 格点特征与拟合窗口异常统计 |
| `neighborhood_tws_features.py` | 邻域与 IDW 特征 |
| `target_stats_features.py`、`target_stats_blockfrozen_features.py` | 目标统计与冻结特征 |
| `gpu_nightly_runner.py`、`gpu_dualhead_runner.py`、`gpu_horizon_runner.py` | 最终脚本导入的共享训练工具；文件名沿用旧名 |

## 回归验证依赖：8 个模块

保留这些模块是为了继续运行现有回归测试，验证信息边界、稀疏日历和历史目标变换。它们不是最终提交的额外训练步骤。

- `block_centered_target.py`、`block_robust_normalized_target.py`
- `gpu_catboost_idw_era5_pattern.py`、`gpu_lgbm_horizon.py`
- `gpu_sparse_replay.py`、`sparse_replay.py`、`sparse_model_features.py`
- `inference_ablation.py`

直接运行 `python -m pytest -q` 验证即可。历史诊断命令可能还需要旧 manifest 和未发布的数据缓存，不应当作新的完整训练入口。

旧候选混合脚本、自动调度、监控同步、Torch 实验及其他变体已移出当前发布范围。本机归档位于被 Git 忽略的 `workspace/archives/`，原始版本可在 Git 提交 `fa835be` 查阅。

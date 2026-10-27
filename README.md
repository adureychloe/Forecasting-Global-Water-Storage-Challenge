# Forecasting Global Water Storage

Zindi 题目：[A Step Ahead of Drought: Forecasting Global Water Storage Challenge](https://zindi.africa)。在全球格点上预测下一个月的陆地水储量（TWS）。

账号 `jinneng`。选定提交 **KJhqz4jM**，公开榜 RMSE **0.6955**。文件是 `submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv`。这是我们提交里私下榜最好的一份，并与当时的公开榜榜首同分。

## 方案

模型不直接吃经纬度。坐标只用来对齐格点和邻域。特征来自官方表的过去值、250 km 反距离加权邻域 TWS、只在拟合窗口上冻结的目标统计，以及按格点-月份拼上的 ERA5 月平均（降水、蒸发、径流、雪、气温、土壤水）。不用 GRACE 或其他 TWS 代用产品。

2010 年以后的数据做全量拟合。五个按最近一次观测 TWS 的年龄分开的头：已观测、gap1、gap2、gap3–6、gap7+。测试集上没有 gap7+。

测试推断用时间对齐的逐步行走：`forecast(t+1)` 只写回 t+1 那个月。最后把 CatBoost、LightGBM、XGBoost 的 aligned 结果按固定权重嵌套混合。完整权重和数据流在 `review/zindi_code_review/DOCUMENTATION.md`。

参与最终混合的六个种子：CatBoost Ordered（l2=5）77 / 2027，LightGBM 77 / 2027 / 123，XGBoost CUDA 77。

## 仓库里有什么

训练脚本留在仓库根目录。它们互相按文件名导入，移动目录会拆掉这条链。

| 路径 | 内容 |
|---|---|
| `review/zindi_code_review/` | 审查包：说明、训练命令、`reproduce.py` |
| `review/zindi_code_review/sources/` | 复现选定文件所需的 8 个组件预测 |
| `review/zindi_code_review/expected/` | 选定提交，SHA256 `c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84` |
| `gpu_*aligned*.py` | 最终 aligned 家族的训练和测试行走 |
| `blend_*.py` | 对着当时榜首做的低权重混合 |
| `experiment_ledger.md` | 从第一个基线开始的实验记录 |
| `SHORTLIST.md` | 2026-09-13 的公开榜路径 |
| `STATE.md` | 同期实验状态。后半段是更早的过程记录 |
| `baseline.py` | 最早的本地树模型基线 |

## 不在 Git 里的内容

官方 `Train.csv` / `Test.csv` / `SampleSubmission.csv`、ERA5 NetCDF、特征缓存、全部历史候选 CSV、压缩包和日志都留在本机。`.gitignore` 排除了 `data/`、`submission/`、`results/` 和各类归档。克隆这个仓库不能单独重新训练，需要自己准备官方数据和 ERA5。

## 复现选定提交

把官方 `SampleSubmission.csv` 放到 `data/SampleSubmission.csv`，然后：

```text
pip install -r requirements.txt
python review/zindi_code_review/reproduce.py
```

脚本用 `sources/` 里的组件预测做嵌套混合，写出 `review/zindi_code_review/output/` 下的选定文件，并与 `expected/` 对照。这一步不训练模型，CPU 上不到一分钟。

## 重新训练

在一台 NVIDIA RTX 4090 上，每个 aligned 种子大约 2.5 小时。从 ERA5 下载到六次行走大约 18–20 小时。命令、输出文件名和“日志出现 `stage=complete` 后要停掉进程”的说明在 `review/zindi_code_review/TRAIN.md`。ERA5 需要本机的 CDS API 配置（`~/.cdsapirc`），密钥不在仓库里。

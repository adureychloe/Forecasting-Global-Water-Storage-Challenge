# Forecasting Global Water Storage Challenge

全球陆地水储量（Terrestrial Water Storage，TWS）月度预测项目，记录 Zindi **A Step Ahead of Drought: Forecasting Global Water Storage Challenge** 的特征工程、模型实验、时间对齐推断和最终提交复现流程。

最终方案结合官方历史观测、空间邻域与 ERA5 气象变量，使用 CatBoost、LightGBM、XGBoost 按观测缺失时长分别建模，再对逐月预测结果进行嵌套混合。

- 项目仓库：[adureychloe/Forecasting-Global-Water-Storage-Challenge](https://github.com/adureychloe/Forecasting-Global-Water-Storage-Challenge)
- 比赛平台：[Zindi](https://zindi.africa)
- 历史选定提交：`KJhqz4jM`，公开榜 RMSE **0.6955**（越低越好）
- 结果来源：[审查说明](review/zindi_code_review/DOCUMENTATION.md)。该文件在本项目提交中私榜表现最好；仓库没有记录私榜具体分数。

## 从哪里开始

| 目的 | 入口 | 所需资源 |
| --- | --- | --- |
| 重建选定提交 CSV | [快速复现](#快速复现选定提交) | 官方提交模板、仓库内组件预测、CPU |
| 理解最终方法与混合权重 | [DOCUMENTATION.md](review/zindi_code_review/DOCUMENTATION.md) | 无需训练 |
| 重新训练 aligned 模型 | [TRAIN.md](review/zindi_code_review/TRAIN.md) | 官方数据、ERA5、支持相应后端的 GPU 环境 |
| 浏览代码与维护约定 | [脚本索引](scripts/README.md)、[结构说明](docs/PROJECT_STRUCTURE.md) | 无需训练 |

## 目录结构

```text
.
├── README.md
├── requirements-reproduce.txt    # 重建 CSV 的最小依赖，固定 NumPy / pandas 版本
├── requirements.txt              # 树模型训练与 ERA5 处理依赖
├── requirements-dev.txt          # 训练依赖 + pytest
├── pytest.ini                    # 测试发现范围与脚本导入路径
├── scripts/                      # 17 个最终方案模块 + 8 个回归验证依赖
├── tests/                        # 自动化回归测试
├── docs/
│   └── PROJECT_STRUCTURE.md       # 精简范围与维护说明
└── review/
    ├── make_review_zip.ps1        # 生成独立审查包
    └── zindi_code_review/
        ├── README.md
        ├── DOCUMENTATION.md      # 最终方法与权重说明
        ├── TRAIN.md
        ├── reproduce.py          # 从已保存的组件预测重建最终 CSV
        ├── sources/              # 8 份混合输入
        └── expected/             # 选定提交的参考文件
```

`data/`、`results/`、`submission/`、`logs/` 和复现生成的 `review/zindi_code_review/output/` 是本地工作目录，均不纳入版本控制。官方数据、ERA5、模型与训练缓存需要自行准备。

`scripts/` 保留同级模块布局，最终模型通过共享 runner 配置训练。运行方式统一为在仓库根目录执行 `python scripts/<脚本名>.py`。旧实验变体、远程部署配置、watcher、私人上传记录和过程笔记不再随仓库发布；本机保留归档，旧版本也可通过 Git 历史查阅。保留范围见 [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

## 快速复现选定提交

这条路径使用已保存的模型预测做混合，不重新训练模型，也不需要 GPU 或 ERA5。

**1. 克隆仓库并创建 Python 3.12 环境。**

```bash
git clone https://github.com/adureychloe/Forecasting-Global-Water-Storage-Challenge.git
cd Forecasting-Global-Water-Storage-Challenge
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

**2. 安装最小依赖，并准备官方模板。**

```bash
python -m pip install -r requirements-reproduce.txt
```

从比赛页面取得官方 `SampleSubmission.csv`，放到 `data/SampleSubmission.csv`。仓库已包含 8 份混合输入和选定提交的参考文件。

**3. 在仓库根目录执行。**

```bash
python review/zindi_code_review/reproduce.py
```

默认生成：

```text
review/zindi_code_review/output/submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv
```

脚本检查 **280,961 行**、官方 ID 顺序、有限预测值，以及嵌套混合与最终一步混合是否一致，并与 `expected/` 中的参考预测比较。参考文件的 SHA256 为：

```text
c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84
```

最小依赖固定为 NumPy 1.26.4、pandas 2.2.2，以匹配参考 CSV 的序列化环境。其他版本可能产生浮点末位或 CSV 字节差异；脚本的数值比较容差为 RMSE ≤ `1e-12`，不要求打印的差异严格为零。

模板或输出也可以另行指定：

```bash
python review/zindi_code_review/reproduce.py --sample /path/to/SampleSubmission.csv --out /path/to/reproduced.csv
```

## 最终方法

1. **构造历史特征**：官方 TWS、SPEI、土壤湿度的历史值与滞后统计；250 km 反距离加权邻域 TWS；仅用拟合窗口构造的冻结目标统计；ERA5 月度变量及其滞后、滚动统计和月度异常。
2. **按观测年龄训练**：用 2010 年及以后的训练数据，分别拟合 `observed`、`gap1`、`gap2`、`gap3_6`、`gap7plus` 五类头。官方测试集没有 `gap7plus` 样本。
3. **按真实日历推断**：按测试月份顺序行走，`forecast(t+1)` 只回填到下一个日历月；不为日历空档凭空生成中间月份。
4. **嵌套混合**：将 aligned 模型预测逐步混入早期基底，再加入 LightGBM seed 123 的 2.5% 预测。精确权重见 [DOCUMENTATION.md](review/zindi_code_review/DOCUMENTATION.md)。

| 模型家族 | 最终 aligned 分量的种子 |
| --- | --- |
| CatBoost Ordered，L2 = 5 | 77、2027 |
| LightGBM | 77、2027、123 |
| XGBoost CUDA | 77 |

最终 aligned 方案移除原始经纬度及其直接编码，坐标用于格点、邻域与 ERA5 对齐；未使用 GRACE 或其他 TWS 代用产品。早期基线与探索实验不等同于最终方案。

## 重新训练

完整数据目录应为：

```text
data/
├── Train.csv
├── Test.csv
├── SampleSubmission.csv
└── era5/
    └── extracted/
        ├── era5_2002_avgua.nc
        ├── era5_2002_avgad.nc
        └── ...                  # 每年一对文件，直到 2018 年
```

安装 `requirements.txt` 后，按 [TRAIN.md](review/zindi_code_review/TRAIN.md) 完成 CDS 配置、ERA5 下载与解包，再运行 `scripts/` 中的四个 aligned 训练入口。普通安装的包不一定支持历史训练使用的 GPU 后端，尤其需要核对 LightGBM 的构建选项。

历史环境为 Linux、Python 3.12、NVIDIA RTX 4090。记录中的每个 aligned 种子耗时约 2.5 小时，完整流程约 18–20 小时，实际取决于硬件、数据准备和脚本种子数量。历史脚本完成后会持续等待；看到 JSON 日志 `"stage": "complete"` 后，用 `Ctrl+C` 或 Supervisor 停止该任务，再继续下一项。

**复现边界：**训练入口重建 aligned 分量；最终提交还依赖已归档的早期 `0.7139` 基底。快速复现使用仓库内保存的分量；用重新训练的分量替换输入时，需要按文档重建整个嵌套混合链，包括最后一步的父文件。GPU 训练结果不承诺逐字节一致。

## 测试与维护

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖历史特征的信息边界、稀疏日历回放、目标变换、监督特征增强，以及入口、数据、缓存与输出路径。测试本身不执行完整 GPU 训练。

仓库聚焦最终方案与可运行的验证代码。实验编号和早期方法比较保留在最终方法说明及部分代码注释中，不代表当前任务或远程进程状态。

审查包可以在 PowerShell 中通过以下命令生成：

```powershell
powershell -File review/make_review_zip.ps1
```

打包细节见 [审查包说明](review/zindi_code_review/README.md)。

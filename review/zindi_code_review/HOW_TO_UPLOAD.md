# 上传说明（给你自己看）

截止日期：**2026-09-25 17:00 GMT**。过时可能取消成绩。

## 表单

https://docs.google.com/forms/d/e/1FAIpQLSe7veHxFdSi4DrY_4wATyPeuq6bnVnCzMwYVgZpj0wmE6zEyg/viewform

建议填写：

- Zindi 用户名：`jinneng`
- 比赛：A Step Ahead of Drought: Forecasting Global Water Storage Challenge
- Best submission ID：`KJhqz4jM`
- 对应文件：`submission_onlineleader_idw_era5_lgbmaligned_extra_seed123_raw_w025.csv`
- Public RMSE：`0.6955`

先读：

- https://zindi.world/learn/documentation-guideline
- https://zindi.world/learn/how-to-ensure-success-when-submitting-your-code-for-review

## 打包

在仓库根目录运行：

```powershell
powershell -File review\make_review_zip.ps1
```

得到 `review/zindi_code_review.zip`。上传这个 zip（或先解压确认后再压）。不要把整个项目（`logs/`、`__pycache__/`、几百个旧 submission）打进去。

zip 内应有：

- `README.md`、`DOCUMENTATION.md`、`TRAIN.md`、`requirements.txt`、`reproduce.py`
- `sources/` 里的 7 个分量 CSV + `expected/` 里的选中提交
- 训练脚本副本（`code/`）

本地先跑：

```powershell
python review\zindi_code_review\reproduce.py
```

应打印 `rms=0.000e+00` 且 SHA256 为 `c2f4c5018bdc0a578c055fa0d2179d3e9785df35f989c13335f01bab9e483b84`。

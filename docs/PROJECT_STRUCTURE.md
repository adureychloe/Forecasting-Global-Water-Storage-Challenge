# 项目结构与维护约定

2026-10-27 整理后，仓库聚焦最终方案、复现输入、方法文档与回归测试。

## 保留范围

| 位置 | 保留内容 |
| --- | --- |
| `scripts/` | 最终训练和推断的 17 个依赖模块，以及现有测试需要的 8 个验证模块 |
| `tests/` | 特征、日历回放、目标变换、增强与路径的回归测试 |
| `review/zindi_code_review/` | 方法说明、数据准备、复现脚本、8 份组件预测和 1 份参考提交 |
| `review/make_review_zip.ps1` | 从归档预测和必要训练模块生成独立审查包 |
| `README.md`、`docs/` | 当前使用与维护说明 |
| `requirements*.txt`、`pytest.ini`、Git 配置 | 依赖、测试发现、忽略规则与跨平台换行约定 |

## 仅在本地归档

以下内容不再随当前仓库版本发布：230 个旧实验脚本、133 份远程配置、54 个启动脚本、watcher 与同步工具、过程笔记、状态快照、私人邮件和上传说明、Starter Notebook 和参考 PDF。本机原文件保存在 `workspace/archives/repository_cleanup_20261027/`，没有删除。原始已发布文件也可从 Git 提交 `fa835be` 恢复。

根目录的旧传输压缩包另存于 `workspace/archives/root_cleanup_20261027/`。官方数据、ERA5、模型、日志、训练缓存与历史候选预测继续保存在本地忽略目录。

其中许多文件原先已经被 Git 跟踪；精简后的变更会从当前版本移除这些文件，仅增加 `.gitignore` 不会解除已有跟踪。这里的移除不清除 Git 历史，也不删除本地归档。

## 运行与导入

- 从仓库根目录运行 `python scripts/<name>.py`。
- 保留共享 runner 的同级导入，避免改动最终模型的配置替换方式。
- 根据 `__file__` 向上定位仓库，`data/`、`results/` 和 `submission/` 仍在根目录。
- `pytest.ini` 仅收集 `tests/` 并加入 `scripts/` 导入路径。
- 历史 manifest 中的源码哈希对应旧版本。需要复查旧实验时，使用原提交，或重新生成参考运行；不要修改哈希绕过完整性校验。
- `review/zindi_code_review/output/` 是复现输出，不再与 `expected/` 重复跟踪。

## 验证

运行 `python -m pytest -q` 检查回归行为，再用官方模板运行 `review/zindi_code_review/reproduce.py`。该脚本应成功比较参考预测；固定 NumPy 1.26.4 / pandas 2.2.2 时生成参考 SHA256。

审查 ZIP 通过 `powershell -File review/make_review_zip.ps1` 生成。它保留完整的 17 个训练模块及其本地导入依赖，支持使用外部官方模板独立重建归档提交。此次整理不重新执行完整 GPU 训练或 ERA5 下载。

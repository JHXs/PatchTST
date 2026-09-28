# Task Packet：基线对比 v2（方向 21）

## Task Packet

- Scope：实现并完成北京 1013 单站点 Informer/TST 公平基线正式矩阵，汇总真实结果并完成合规自检。
- Files to read：`docs/基线对比-v2/00_协议.md`、`01_实现工作单.md`、参考分支训练骨架、三份复用归档及其校验文件。
- Files allowed to edit：工作单列出的四个 Python 文件、`docs/基线对比-v2/02_*` 与 `04_*`、`experiments/results/baseline_v2/`、`tables/baseline_v2/`、`figures/baseline_v2/`，以及本轮计划/审查登记。
- Required skills：paper-orchestration、experiment-results-planning、figures-python、statistical-analysis、verification。
- Evidence/data inputs：北京无泄漏数据构建器；8 个预注册新臂的训练日志与预测；O-locked、GRU/LSTM、传统基线归档。
- Required artifacts：512 身份原始指标；主表、容量曲线、逐 lead、分层、秩相关、双口径、独立复算、十项合规表；PNG/SVG 图；中文结果报告。
- Rejection checks：任何训练语义漂移；超过 8 臂或 512 次；测试集选择/调参；非共同子集配对；失败身份丢失；归档未校验；十项合规未全绿。
- Validation commands：Python 编译；`test_baseline_v2.py` 全套测试；骨架复现门；冒烟；正式矩阵终态计数；汇总重跑；独立复算；`git diff --check`。

## 阶段

当前为 S3 Experiments；正式矩阵与汇总完成后转入 S5 Review。

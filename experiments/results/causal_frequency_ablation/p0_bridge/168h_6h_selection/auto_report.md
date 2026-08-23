# 频域分支 P0 自动实验摘要

> SELECTION DATA - validation only; do not report as final confirmation.

- 任务：168→6
- 评估划分：valid
- 训练期筛选站点数：18
- 目标时间交集为空：True
- Gate F1 状态：passed

## 验证集聚合结果

| 变体 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) |
|---|---:|---:|---:|
| degraded_patchtst_clean | 50.9872 ± 0.5351 | 33.4985 ± 0.4268 | 50.664 ± 3.026 |
| locked_st_clean | 50.3299 ± 0.5707 | 32.9070 ± 0.7131 | 50.437 ± 2.564 |

## 锁定 ST 相对退化基线

- 改善种子数：3/3
- 平均 RMSE 相对改善：1.2891%

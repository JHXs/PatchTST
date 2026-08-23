# 频域分支 P0 自动实验摘要

> SELECTION DATA - validation only; do not report as final confirmation.

- 任务：24→1
- 评估划分：valid
- 训练期筛选站点数：18
- 目标时间交集为空：True
- Gate F1 状态：pending

## 验证集聚合结果

| 变体 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) |
|---|---:|---:|---:|
| degraded_patchtst_clean | 23.7705 ± 0.2346 | 13.0265 ± 0.1589 | 22.271 ± 1.879 |
| locked_st_clean | 22.5771 ± 0.3333 | 12.6884 ± 0.1583 | 21.965 ± 0.906 |

## 锁定 ST 相对退化基线

- 改善种子数：3/3
- 平均 RMSE 相对改善：5.0230%

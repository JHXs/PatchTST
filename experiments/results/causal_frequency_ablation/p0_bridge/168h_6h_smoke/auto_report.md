# 频域分支 P0 自动实验摘要

> SMOKE TEST DATA - DO NOT USE AS THESIS EVIDENCE.

- 任务：168→6
- 评估划分：valid
- 训练期筛选站点数：18
- 目标时间交集为空：True
- Gate F1 状态：pending

## 验证集聚合结果

| 变体 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) |
|---|---:|---:|---:|
| degraded_patchtst_clean | 60.6903 ± 0.0000 | 44.1999 ± 0.0000 | 75.721 ± 0.000 |
| locked_st_clean | 59.6605 ± 0.0000 | 42.7103 ± 0.0000 | 69.964 ± 0.000 |

## 锁定 ST 相对退化基线

- 改善种子数：1/1
- 平均 RMSE 相对改善：1.6967%

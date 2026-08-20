# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：24 小时历史预测未来 1 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6115, 'valid': 873, 'test': 1748}
- 当前评估划分：valid
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 23.7420 ± 0.0000 | 12.8772 ± 0.0000 | 20.935 ± 0.000 | 0.000000 |
| st_pairwise_delta_forecast | 22.6656 ± 0.0000 | 12.7932 ± 0.0000 | 21.718 ± 0.000 | 0.037370 |
| st_sparse_delta_forecast | 22.5624 ± 0.0000 | 12.8725 ± 0.0000 | 22.231 ± 0.000 | 0.043942 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

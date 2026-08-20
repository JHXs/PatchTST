# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：24 小时历史预测未来 1 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6115, 'valid': 873, 'test': 1748}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 21.5158 ± 0.1171 | 10.0619 ± 0.0814 | 15.916 ± 0.298 | 0.000000 |
| st_pairwise_delta_forecast | 20.9357 ± 0.4455 | 10.0632 ± 0.3605 | 17.215 ± 0.612 | 0.045522 |
| st_sparse_delta_forecast | 21.0220 ± 0.5226 | 10.1909 ± 0.3833 | 17.357 ± 0.602 | 0.048020 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

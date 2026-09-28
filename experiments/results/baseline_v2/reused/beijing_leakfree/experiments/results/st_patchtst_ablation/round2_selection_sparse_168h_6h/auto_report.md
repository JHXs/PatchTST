# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：168 小时历史预测未来 6 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6010, 'valid': 859, 'test': 1718}
- 当前评估划分：valid
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 50.5196 ± 0.0000 | 32.4691 ± 0.0000 | 51.564 ± 0.000 | 0.000000 |
| st_pairwise_delta_forecast | 50.1322 ± 0.0000 | 32.2514 ± 0.0000 | 51.127 ± 0.000 | 0.045501 |
| st_sparse_delta_forecast | 50.1220 ± 0.0000 | 32.2388 ± 0.0000 | 51.264 ± 0.000 | 0.048459 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

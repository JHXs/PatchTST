# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：168 小时历史预测未来 24 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 5998, 'valid': 857, 'test': 1714}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 67.6968 ± 2.2815 | 47.7401 ± 1.5954 | 62.664 ± 1.033 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 67.8627 ± 2.4038 | 47.8595 ± 1.6115 | 62.569 ± 1.106 | 0.029573 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

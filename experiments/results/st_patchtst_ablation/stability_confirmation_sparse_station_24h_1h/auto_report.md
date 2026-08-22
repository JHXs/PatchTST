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
| degraded_patchtst | 21.4439 ± 0.2003 | 10.1349 ± 0.0805 | 16.283 ± 0.353 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 20.7539 ± 0.2391 | 9.9902 ± 0.0872 | 16.811 ± 0.259 | 0.049522 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

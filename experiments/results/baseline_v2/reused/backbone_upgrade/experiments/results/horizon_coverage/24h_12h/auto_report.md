# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：24 小时历史预测未来 12 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6107, 'valid': 872, 'test': 1746}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 53.3911 ± 0.2527 | 32.8257 ± 0.2604 | 44.762 ± 0.511 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 51.4755 ± 0.2270 | 31.8911 ± 0.2113 | 43.446 ± 0.250 | 0.158301 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

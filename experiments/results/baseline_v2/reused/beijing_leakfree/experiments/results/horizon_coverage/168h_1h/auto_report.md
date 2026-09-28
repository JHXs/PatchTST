# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：168 小时历史预测未来 1 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6014, 'valid': 859, 'test': 1719}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 22.0166 ± 0.1953 | 11.6897 ± 0.2856 | 21.001 ± 0.794 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 21.7993 ± 0.1821 | 11.5641 ± 0.2474 | 20.601 ± 0.435 | 0.036268 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：48 小时历史预测未来 6 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6094, 'valid': 871, 'test': 1742}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 42.6386 ± 0.2172 | 24.4608 ± 0.2219 | 34.429 ± 0.525 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 41.5331 ± 0.4477 | 24.0009 ± 0.4513 | 35.105 ± 0.606 | 0.086303 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

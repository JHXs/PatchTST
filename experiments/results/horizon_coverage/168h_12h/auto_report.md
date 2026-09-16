# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：168 小时历史预测未来 12 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6006, 'valid': 858, 'test': 1717}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 55.0068 ± 1.0562 | 36.2734 ± 1.0532 | 50.991 ± 1.507 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 55.0581 ± 1.2378 | 36.3082 ± 1.1814 | 50.819 ± 1.391 | 0.028234 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

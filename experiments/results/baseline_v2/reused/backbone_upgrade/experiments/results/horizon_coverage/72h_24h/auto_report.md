# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：72 小时历史预测未来 24 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6065, 'valid': 866, 'test': 1734}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 67.1383 ± 1.0153 | 45.8206 ± 0.8523 | 58.223 ± 1.119 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 67.2669 ± 0.9751 | 46.0167 ± 0.7770 | 57.775 ± 0.957 | 0.064992 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

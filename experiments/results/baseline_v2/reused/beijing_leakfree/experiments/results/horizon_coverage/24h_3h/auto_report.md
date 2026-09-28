# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：24 小时历史预测未来 3 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6113, 'valid': 874, 'test': 1747}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 31.6844 ± 0.0996 | 16.3923 ± 0.0809 | 24.347 ± 0.349 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 30.7773 ± 0.0861 | 16.0831 ± 0.1403 | 24.674 ± 0.314 | 0.072433 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

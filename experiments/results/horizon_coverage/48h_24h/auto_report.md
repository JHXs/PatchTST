# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：48 小时历史预测未来 24 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6082, 'valid': 869, 'test': 1738}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 71.1568 ± 1.4933 | 47.6252 ± 0.6045 | 59.486 ± 0.471 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 69.4430 ± 1.5500 | 46.6521 ± 0.7241 | 58.728 ± 0.901 | 0.148360 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

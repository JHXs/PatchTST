# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：24 小时历史预测未来 1 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6115, 'valid': 873, 'test': 1748}
- 当前评估划分：valid
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 23.8719 ± 0.1221 | 12.9857 ± 0.1117 | 21.676 ± 0.666 | 0.000000 |
| st_station_bias_delta_forecast | 22.8740 ± 0.1249 | 12.8392 ± 0.0663 | 22.647 ± 0.820 | 0.041481 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：168 小时历史预测未来 6 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6010, 'valid': 859, 'test': 1718}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 42.6285 ± 0.8874 | 26.3675 ± 0.8442 | 40.112 ± 1.431 | 0.000000 |
| st_station_bias_delta_forecast | 42.1740 ± 0.8998 | 26.1704 ± 0.8461 | 39.713 ± 1.006 | 0.081418 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

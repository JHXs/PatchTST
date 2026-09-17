# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：72 小时历史预测未来 12 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6073, 'valid': 868, 'test': 1736}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 54.9511 ± 1.2978 | 34.8322 ± 0.6213 | 46.676 ± 0.388 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 54.4564 ± 1.0967 | 34.5468 ± 0.6275 | 46.540 ± 0.586 | 0.069557 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

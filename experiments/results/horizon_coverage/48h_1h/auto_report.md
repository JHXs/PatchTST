# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：48 小时历史预测未来 1 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6098, 'valid': 871, 'test': 1743}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 21.6320 ± 0.1222 | 10.6501 ± 0.0683 | 17.628 ± 0.213 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 20.9051 ± 0.1741 | 10.3430 ± 0.0966 | 17.873 ± 0.343 | 0.044497 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

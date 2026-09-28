# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：168 小时历史预测未来 3 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6013, 'valid': 858, 'test': 1719}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 32.5696 ± 0.2139 | 18.9264 ± 0.2555 | 31.006 ± 0.505 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 32.1241 ± 0.2675 | 18.4674 ± 0.1434 | 30.449 ± 0.976 | 0.053809 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

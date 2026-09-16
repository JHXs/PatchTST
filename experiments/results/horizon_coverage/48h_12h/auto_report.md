# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：48 小时历史预测未来 12 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6090, 'valid': 870, 'test': 1741}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 56.8216 ± 0.9770 | 35.1177 ± 0.4682 | 46.447 ± 0.573 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 55.8383 ± 0.8473 | 34.7260 ± 0.3373 | 46.443 ± 0.533 | 0.089161 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

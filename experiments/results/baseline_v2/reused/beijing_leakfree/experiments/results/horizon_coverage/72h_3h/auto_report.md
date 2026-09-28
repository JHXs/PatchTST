# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：72 小时历史预测未来 3 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6080, 'valid': 868, 'test': 1738}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 32.1307 ± 0.5186 | 17.4093 ± 0.2581 | 26.831 ± 0.634 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 31.4972 ± 0.5617 | 17.1346 ± 0.3790 | 27.019 ± 0.627 | 0.057035 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

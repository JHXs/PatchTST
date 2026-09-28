# ST-PatchTST 自动实验摘要

> REAL EXPERIMENT DATA - generated from the recorded runs.

- 任务：72 小时历史预测未来 1 小时 PM2.5
- 中心站：1013
- 站点数：19
- 样本划分：{'train': 6081, 'valid': 869, 'test': 1738}
- 当前评估划分：test
- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。

## 聚合结果

| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |
|---|---:|---:|---:|---:|
| degraded_patchtst | 21.7771 ± 0.4430 | 10.8941 ± 0.1684 | 18.429 ± 0.842 | 0.000000 |
| st_sparse_station_bias_delta_forecast | 21.3425 ± 0.6143 | 10.7920 ± 0.2199 | 18.496 ± 0.501 | 0.032442 |

## 相对退化基线的配对变化

负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。

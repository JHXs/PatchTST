# 图形数据清单

| Figure candidate | Data file | Status | Notes |
|---|---|---|---|
| 各模型逐种子 RMSE | `experiments/results/st_patchtst_ablation/*/raw_metrics.csv` | 数据已生成，图待绘制 | 真实实验数据 |
| 训练/验证损失曲线 | `experiments/results/st_patchtst_ablation/*/training_logs/*.csv` | 数据已生成，图待绘制 | 真实实验数据 |
| 邻站扰动前后误差 | `experiments/results/st_patchtst_ablation/*/raw_metrics.csv` | 数据已生成，图待绘制 | 真实实验数据 |
| 六个预测步误差变化 | `round2_station_bias_168h_6h/per_horizon_relative_summary.csv` | 数据已生成，图待绘制 | 可画分组折线图，突出均值而非单次最优 |
| 邻站平均门控权重 | `round2_station_bias_*/station_gate_aggregate.csv` | 数据已生成，图待绘制 | 可与滞后诊断的线性增益排序并列 |
| 候选传播滞后诊断 | `lag_diagnostics/lag_cv_summary.csv`、`lag_correlation_summary.csv` | 数据已生成，图待绘制 | 同步 0 h 为主，其余滞后作为负结果 |

本轮优先生成数据和报告，不在缺乏稳定结果时预制结论性图形。

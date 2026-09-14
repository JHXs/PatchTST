# 图形数据清单

| Figure candidate | Data file | Status | Notes |
|---|---|---|---|
| 各模型逐种子 RMSE | `experiments/results/st_patchtst_ablation/*/raw_metrics.csv` | 数据已生成，图待绘制 | 真实实验数据 |
| 训练/验证损失曲线 | `experiments/results/st_patchtst_ablation/*/training_logs/*.csv` | 数据已生成，图待绘制 | 真实实验数据 |
| 邻站扰动前后误差 | `experiments/results/st_patchtst_ablation/*/raw_metrics.csv` | 数据已生成，图待绘制 | 真实实验数据 |
| 六个预测步误差变化 | `round2_station_bias_168h_6h/per_horizon_relative_summary.csv` | 数据已生成，图待绘制 | 可画分组折线图，突出均值而非单次最优 |
| 邻站平均门控权重 | `round2_station_bias_*/station_gate_aggregate.csv` | 数据已生成，图待绘制 | 可与滞后诊断的线性增益排序并列 |
| 候选传播滞后诊断 | `lag_diagnostics/lag_cv_summary.csv`、`lag_correlation_summary.csv` | 数据已生成，图待绘制 | 同步 0 h 为主，其余滞后作为负结果 |
| 五轮稳定性确认 | `stability_confirmation_paired.csv`、`stability_confirmation_summary.csv` | 数据已生成，图待绘制 | 必须同时展示前四轮失败，禁止只画最终有利结果 |
| Top-k验证选择 | `topk_validation_selection.csv`、`topk_validation_scores.csv` | 数据已生成，图待绘制 | 标为验证集结构选择，不与最终确认混画 |

本轮优先生成数据和报告；最终确认已通过严格门，但尚未生成论文图。

## Round 14：跨城市泛化确认（广州，已完成）

| Figure | Data file | Status | Notes |
|---|---|---|---|
| 逐站配对降幅 | `tables/round14/T2_per_station.csv` | 已绘制 `figures/round14/F1_per_station_reduction.pdf` | 8 个未消费中心站 × 5 种子 |
| 逐预测步误差 | `tables/round14/T3_per_lead.csv` | 已绘制 `figures/round14/F2_per_lead.pdf` | 168→6 六步 |
| 逐块稳健性 | `tables/round14/T4_per_block.csv` | 已绘制 `figures/round14/F3_per_block.pdf` | 5 个确认块 |
| 配对分布 | `tables/round14/T2_per_station.csv`（配对来源 `cross_city_generalization_summary/paired_effects.csv`） | 已绘制 `figures/round14/F4_pair_distribution.pdf` | 每任务 40 配对 |
| 逐站 RMSE 散点 | `tables/round14/T2_per_station.csv` | 已绘制 `figures/round14/F5_rmse_scatter.pdf` | 对角线上方为改善 |
| SMAPE 分位分解 | `tables/round14/T7_smape_quantile_decomposition.csv` | 已绘制 `figures/round14/F6_smape_quantiles.pdf` | 训练段分位 |
| 高浓度子集 | `tables/round14/T5_peak_subset.csv` | 已绘制 `figures/round14/F7_peak_subset.pdf` | 各站训练段 q90 |
| 示例时序 | 归档 `round14-cross-city-generalization-71bb1d9.tar.zst` 内预测 npz | 已绘制 `figures/round14/F8_example_series_9033_24x1.pdf` | 含空间修正量面板 |

生成脚本：`make_round14_paper_artifacts.py`；说明见 `figures/round14/README.md`。

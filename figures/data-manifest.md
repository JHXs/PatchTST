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

## 北京 1013 第一创新（已绘制，2026-09-14）

| Figure | Data file | Status | Notes |
|---|---|---|---|
| 最终确认逐种子降幅 | `experiments/results/st_patchtst_ablation/stability_confirmation_paired.csv` | 已绘制 `figures/beijing/BF1_final_per_seed.pdf` | 种子 2047–2051 |
| 五轮尝试演进（含失败轮） | `stability_confirmation_summary.csv` | 已绘制 `figures/beijing/BF2_attempts.pdf` | 按严格门着色，标注改善种子数 |
| 邻站干预消融 | `stability_confirmation_topk5_*/raw_metrics.csv` | 已绘制 `figures/beijing/BF3_interventions.pdf` | 禁用/置零/打乱/中心复制 |
| Top-k 验证选择 | `topk_validation_selection.csv`、`topk_validation_scores.csv` | 已绘制 `figures/beijing/BF4_topk_selection.pdf` | 2024–2026 验证段 |
| 传播滞后诊断 | `lag_diagnostics/lag_cv_summary.csv` | 已绘制 `figures/beijing/BF5_lag_diagnostics.pdf` | 负结果 |
| 训练/验证损失 | `stability_confirmation_topk5_*/training_logs/*.csv` | 已绘制 `figures/beijing/BF6_training_curves.pdf` | 5 种子 |

生成脚本：`make_beijing_paper_artifacts.py`；说明见 `figures/beijing/README.md`。

| 匹配组件消融（两种协议） | `experiments/results/frozen_protocol_component_ablation/*/raw_metrics.csv` | 已绘制 `figures/beijing/BF7_structure_ablation.pdf` | test 划分、种子 2024–2026；valid 选择轮见 B9 |

| 示例时序（北京） | 重跑 predictions（归档 `beijing-example-rerun-2047-2051.tar.zst`） | 已绘制 `figures/beijing/BF8_example_series_1013_24x1.pdf`、`BF8_example_series_1013_168x6_lead1.pdf` | 按同配置重跑，5 种子逐位复现 |

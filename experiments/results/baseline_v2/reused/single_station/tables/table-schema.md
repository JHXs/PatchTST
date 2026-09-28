# 表格数据规范

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| 第一轮主结果表 | 比较输入端空间融合 | 模型 | RMSE/MAE/SMAPE、mean ± std | `combined_summary.csv` | 实验运行器 |
| 第二轮主结果表 | 比较预测端密集、稀疏和站点偏置模型 | 任务×模型 | RMSE/MAE/SMAPE、配对变化、改善种子数 | `round2_combined_summary.csv` | 汇总脚本 |
| 配对改进表 | 比较相同种子差值 | ST模型×种子 | ST-baseline 指标差 | `paired_differences.csv` | 实验运行器 |
| 逐预测步表 | 判断空间收益随预测距离的变化 | 任务×模型×预测步 | RMSE/MAE、相对变化 | `round2_station_bias_*/per_horizon_relative_summary.csv` | 汇总脚本 |
| 空间诊断表 | 判断空间分支是否生效 | 模型×种子 | alpha、残差比、熵、null权重 | `raw_metrics.csv` | 实验运行器 |
| 真实性消融表 | 检查邻站依赖 | 模型×干预 | 正常/显式禁用/置零/全局打乱/中心复制 RMSE | `raw_metrics.csv` | 实验运行器 |
| 站点门控表 | 检查站点身份先验与动态门控 | 任务×站点 | 偏置、门控均值及跨种子标准差 | `station_gate_aggregate.csv` | 汇总脚本 |
| 滞后诊断表 | 判断是否应显式移动邻站序列 | 站点×滞后 | 原值/差分相关、扩展窗口 RMSE 增益 | `lag_diagnostics/*.csv` | 滞后诊断脚本 |
| Top-k验证选择表 | 固定最终稀疏邻站数 | k×任务 | 配对RMSE降幅、改善种子数、归一化门槛分数 | `topk_validation_selection.csv`、`topk_validation_scores.csv` | Top-k汇总脚本 |
| 稳定性确认表 | 区分探索结果、方向一致性和严格放行 | 确认策略×任务 | 配对RMSE降幅、Bootstrap CI、改善种子数、精确符号检验、门槛状态 | `stability_confirmation_*.csv`、`stability_release_status.json` | 稳定性汇总脚本 |

## 北京 1013 第一创新论文表（2026-09-14）

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| `B1_final_gates` | 最终确认与预注册门 | 任务 | base/候选 RMSE、CI95、改善种子数、符号 p、门槛、严格门 | `stability_confirmation_summary.csv` | `make_beijing_paper_artifacts.py` |
| `B2_final_per_seed` | 逐种子配对 | 任务×种子 | RMSE/MAE 配对与降幅 | `stability_confirmation_paired.csv` | 同上 |
| `B3_all_attempts` | 五轮尝试（含失败） | 尝试×任务 | 均值/标准差、种子计数、三类门状态 | 同上 | 同上 |
| `B4_neighbor_interventions` | 邻站干预消融 | 任务×变体 | 正常/禁用/置零/打乱/中心复制的 RMSE 与相对变化 | `stability_confirmation_topk5_*/raw_metrics.csv` | 同上 |
| `B5_gate_diagnostics` | 空间分支诊断 | 任务×变体 | alpha、残差比例、门控熵、空邻站权重等 | `stability_confirmation_topk5_*/summary_metrics.csv` | 同上 |
| `B6_topk_validation`、`B6b_topk_scores` | Top-k 验证选择 | k×任务 | 平均降幅、改善种子数、归一化余量 | `topk_validation_selection.csv` | 同上 |
| `B7_lag_diagnostics` | 滞后诊断 | 滞后×站点聚合 | 增益百分比与改善站点数 | `lag_diagnostics/lag_cv_summary.csv` | 同上 |
| `B8_architecture_ablation` | 匹配组件消融（test 划分） | 协议×变体×任务 | RMSE、配对降幅、改善种子数、邻站干预增幅 | `experiments/results/frozen_protocol_component_ablation/` | `make_beijing_paper_artifacts.py` |
| `B9_dev_selection_validation` | 开发选择与 k 扫描（valid 划分） | 变体×任务 | 验证划分的 RMSE 与配对降幅 | `round3..round6_selection_*` | 同上 |
| `B10_reproducibility_rerun_vs_recorded` | 重跑一致性核对 | 任务×变体×种子 | 重跑/记录 RMSE、MAE、best_epoch 与差值 | 重跑目录 + `stability_confirmation_topk5_*` | `make_beijing_paper_artifacts.py --example-*-dir` |

## Round 14 论文表（跨城市泛化确认）

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| `T1_main_gates` | 预注册 G1–G4 判定 | 任务 | 池效应、门阈值、配对/站/块计数、合取判定 | `cross_city_generalization_summary/gate_summary.json` | `make_round14_paper_artifacts.py` |
| `T2_per_station` | 逐站主结果 | 任务×站 | base/ST 的 RMSE/MAE/SMAPE、降幅 | `station_metrics.csv` | 同上 |
| `T3_per_lead` | 逐预测步 | 任务×lead | RMSE/MAE/SMAPE 与降幅 | `lead_metrics.csv` | 同上 |
| `T4_per_block` | 逐确认块 | 任务×块 | 两臂 RMSE 与降幅 | `block_metrics.csv` | 同上 |
| `T5_peak_subset` | 高浓度子集 | 任务×站 | q90 子集 RMSE/MAE、覆盖数 | `peak_metrics.csv` | 同上 |
| `T6_headline_metrics` | 池化主指标 | 任务×臂 | 池化 RMSE/MAE/SMAPE、元素数 | `run_metrics.csv` | 同上 |
| `T7_smape_quantile_decomposition` | SMAPE 分位分解 | 任务×分位 | base/ST SMAPE 与变化 | `quantile_smape_decomposition.csv` | 同上 |
| `T8_top5_neighbours` | 筛站结果 | 站 | Top-5 邻站与相关性 | 归档 `run_metadata.json` | 同上 |
| `T9_data_coverage` | 数据覆盖 | 站 | 共同行数与三段行数、候选数 | 归档 `run_metadata.json` | 同上 |

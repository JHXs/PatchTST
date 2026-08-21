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

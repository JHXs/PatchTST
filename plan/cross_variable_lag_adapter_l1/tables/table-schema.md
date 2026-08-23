# L1 表格契约

| Table/artifact | Purpose | Rows | Metrics/fields | Data source | Aggregation |
|---|---|---|---|---|---|
| raw_metrics.csv | 每任务/种子/变体总指标 | variant×seed | RMSE/MAE/SMAPE/bias、epoch/hash | validation_predictions.csv | sample×horizon |
| horizon_metrics.csv | 逐预测步 | variant×seed×h | RMSE/MAE/bias | validation_predictions.csv | 每 h |
| scenario_metrics.csv | 场景安全 | variant×seed×scenario | n/RMSE/MAE/bias | predictions + train thresholds | 场景单元格 |
| factorial_metrics.csv | 2×2 | seed×metric | 四格误差、四增益、I、R_attn | recomputed metrics | 逐种子 |
| paired_metrics.csv | 硬门比较 | task×candidate×baseline×seed | 绝对/相对改善 | recomputed metrics | 逐种子及均值 |
| shuffle_metrics.csv | 对齐依赖 | train seed×perm seed | G_orig/G_shuffle/L_s | fixed diagnostic subset | 不跨训练种子合并 |
| patchtst_state_audit.csv | F10冻结身份 | base variant×seed×stage/epoch | 持久/完整state hash、参数buffer数、差异张量、eval状态 | 现场模型与重新读取checkpoint | 不聚合，逐阶段硬断言 |
| run_manifest.json | 可追溯性 | run | Git commit/dirty/porcelain、关键源码SHA256、smoke标识 | Git与源码文件 | 不聚合 |
| resource_smoke.json | 168→6 GPU资源事实 | 一个真实batch512 | 设备、形状、峰值显存、梯度、冻结状态 | F11完整前向/反向 | 不作性能解释 |
| gate_status.json | L1 硬门 | gate | observed/pass | preceding tables | 仅正式两任务三种子可判定 |

任何 `_smoke` 表均是流程数据，不得复制到论文结果表。

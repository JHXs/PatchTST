# 第二创新点：频域分支表格数据规范

> 当前均为计划表，不含模拟数字。所有数值必须来自 `experiments/results/causal_frequency_ablation/` 的真实 CSV/JSON。

| Table | Purpose | Rows | Required metrics | Data source | Status |
|---|---|---|---|---|---|
| F0 无泄漏过渡表 | 验证新协议下 ST 仍成立 | 任务×模型×种子 | RMSE/MAE/SMAPE、站点数、样本数、目标交集 | `p0_bridge/*/raw_metrics.csv`、`dataset_metadata.json` | selection数据已生成 |
| F1 频率特异性主表 | 区分频域结构与新增容量 | ST/时域适配器/固定频带×种子 | 配对 RMSE/MAE、参数、时间、通过门 | `p1_fixed_selection/*` | selection数据已生成，F3失败 |
| FA1 唯一替代表 | 检验全局FFT的时间定位限制 | ST/时域适配器/因果局部滤波器×种子 | 配对RMSE/MAE、参数、重构与门状态 | `p1_alternative_selection/*` | selection数据已生成，FA1失败 |
| F2 可学习频带表 | 检验受约束频带学习 | 固定/可学习×种子 | 配对误差、边界周期、重叠率、塌缩状态 | `p2_learnable_selection/*` | 未进入 |
| F3 动态门控表 | 区分全局、自条件和气象条件化 | 变体×种子 | 误差、门控均值/方差/熵、置乱差值 | `p3_gating_selection/*` | 未进入 |
| 气象控制表 | 排除气象只是额外输入容量 | 自条件/气象残差/气象门控×种子 | 参数匹配、配对 RMSE/MAE、门控诊断 | `p3_gating_selection/*` | 未进入 |
| 峰值结果表 | 检验峰值过度平滑 | MSE/峰值目标×种子×分组 | 整体/q90/非q90 RMSE/MAE、P/R/F1 | `p4_peak_selection/*/peak_metrics.csv` | 未进入 |
| 逐预测步表 | 判断收益随 horizon 的变化 | 变体×预测步 | RMSE/MAE及配对变化 | `*/per_horizon_metrics.csv` | 计划 |
| 真实性干预表 | 确认频域残差和门控被使用 | 变体×干预 | 正常/禁用/固定/置乱/单带 RMSE | `*/frequency_diagnostics/`、`raw_metrics.csv` | 计划 |
| 效率与 VMD 表 | 比较研究价值与计算成本 | rFFT/VMD×任务 | 参数、训练/推理/分解/缓存时间、显存 | `p5_vmd/*` 及对应 rFFT 目录 | 未进入 |
| 内部确认表 | 报告锁定结构的稳定性 | 任务×种子 | 配对效应、5/5、符号检验、块Bootstrap | `p6_internal_confirmation/*` | 未进入 |
| 外部泛化表 | 界定站点/城市适用范围 | 数据集×模型×种子 | RMSE/MAE、配对变化、方向一致性 | `p6_external_generalization/*` | 未进入 |

论文主表优先保留 F1 频率特异性、锁定结构内部确认和外部泛化；其余进入消融或附录。选择集与确认集不得合并计算均值。

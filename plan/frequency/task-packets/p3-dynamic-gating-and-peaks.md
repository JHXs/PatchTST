# P3/P4 任务包：动态门控、气象控制与峰值感知

> 状态：`not_entered`。FA1失败后本任务包仅保留历史规划，不得执行。

## 进入条件

原进入条件为Gate F4通过，或F3通过后回退固定频带；实际F3/FA1均失败，不满足进入条件。

## 任务目标

先检验中心 PM2.5 状态条件化的频带门控，再检验气象信息是否提供超出普通气象残差的增量，最后单独检验峰值感知损失。三个问题不得一次打包成一个候选。

## P3a：自条件门控

- 门控上下文只来自中心站 PM2.5 历史摘要。
- 与全局频带权重使用相同适配器、同检查点和训练预算。
- 必报门控样本标准差、熵、污染水平分组权重和 `fixed_uniform_gates`/`permute_gate_context` 干预。

## P3b：气象增量

- 输入仅限协议预定义的中心站气象特征。
- 必须同时训练 `st_meteorology_residual_control`，其参数量与气象门控新增容量匹配。
- 气象门控必须同时超过自条件门控和气象普通残差，才允许表述为“气象条件化频带选择”；不得写成气象对污染的因果效应。

## P4：峰值感知

- 只在 P3 锁定结构上比较普通 MSE 与峰值目标。
- q90 只由 train 目标计算并冻结。
- 必报整体、q90、非 q90、逐预测步误差和超阈值 precision/recall/F1。
- q90 RMSE/MAE不得同时退化，至少一项改善 1%，整体 RMSE 退化小于 0.2%，方可通过 F6。

## 选择与停止

- 全程只使用 2052–2054 验证种子。
- 每个子阶段只允许胜出者进入下一阶段；失败模块删除对应机理声称，不能与后续模块联合后重新挑选。
- `permute_gate_context` 和单频带干预仅作真实性诊断，不参与选模。

## 输出位置

- `experiments/results/causal_frequency_ablation/p3_gating_selection/168h_6h/`
- `experiments/results/causal_frequency_ablation/p4_peak_selection/168h_6h/`

# P1 任务包：固定频带 MVP 与等容量对照

## 进入条件

Gate F1 已通过，且 P0 的锁定 ST 检查点、数据元数据和验证种子结果完整。

## 任务目标

实现可微的窗口内 `rfft → 固定平滑频带 → irfft` 预测端残差，并用等容量时域适配器检验收益是否具有频率特异性。

## 锁定配置

- 主任务 168→6，辅助任务 24→1。
- 三个频带：周期大于 48 h、12–48 h、小于 12 h；仅在 168 h 主任务解释三尺度。
- 频域输出层零初始化，`beta_max=0.5`；禁用频域时必须逐元素恢复锁定 ST。
- P1 采用固定等权融合；频带权重学习留到 P2。
- `st_time_residual_adapter` 与频域分支共享隐层宽度、池化、输出层、dropout、训练预算和残差上限；可训练参数差异不超过 5%。

## 必需实现与测试

1. rFFT/irFFT 重构、奇偶长度、批尺寸 1、CPU/ROCm 前向和反向测试。
2. 频带掩码覆盖所有频点、逐频率和为 1、平滑过渡和有限梯度测试。
3. 零初始化、`disable_frequency`、冻结 PatchTST/ST 参数及仅适配器更新测试。
4. `locked_st_clean`、时域容量控制和固定频带使用同种子同检查点的配对运行。
5. 参数量、训练/推理时间、频带能量和残差 RMS 诊断。

## 选择与停止

- 仅使用 2052–2054 验证种子。
- 固定频带需在 168→6 上 3/3 优于锁定 ST、平均降幅至少 0.5%，且平均 RMSE 严格优于等容量时域适配器。
- 未通过则 Gate F3 失败，停止 P2–P5；结果作为“频率分解未超过新增容量”的负结果保留。

## 输出位置

`experiments/results/causal_frequency_ablation/p1_fixed_selection/<history>h_<horizon>h/`

## 最小验证

```bash
uv run python -m py_compile *.py
uv run python -m unittest -v test_st_patchtst.py
# 实现后补充频域模块与运行器测试命令
git diff --check
```

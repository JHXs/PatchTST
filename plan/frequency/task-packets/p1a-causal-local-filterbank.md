# P1A 任务包：唯一替代——因果局部多尺度滤波器组

## 进入原因

固定全窗口rFFT在F3中虽3/3略优于锁定ST，但平均改善仅0.3502%，且3/3不如等容量时域适配器。根据预注册止损条款，只允许一个替代分解，不再扫描FFT边界或门控。

> 执行状态（2026-08-23）：已完成，FA1失败并触发停止条款。

## 锁定结构

对中心PM2.5历史窗口计算只读过去的尾随均值：

```text
ma12 = trailing_mean(x, 12)
ma48 = trailing_mean(x, 48)
low  = ma48
mid  = ma12 - ma48
high = x - ma12
```

三路逐点严格重构原序列。左边界只复制窗口首个已观测值，不使用右侧或预测起点后的值。卷积核固定，不学习、不扫描；12/48小时在查看替代结果前锁定。

## 公平对照

- 变体ID：`st_causal_filterbank_residual`；
- 主对照：同种子 `locked_st_clean`；
- 容量对照：`st_time_residual_adapter`；
- 复用相同P0锁定ST检查点、2052–2054种子、训练预算、编码器、池化、输出头、beta和最佳验证损失规则；
- 可训练参数必须与时域控制完全相同。

## 放行与停止

FA1通过必须同时满足：168→6上3/3优于锁定ST、平均RMSE改善≥0.5%、平均RMSE严格优于时域控制、严格退化/重构/梯度/冻结检查通过。

若任一条件失败：停止当前频域/多尺度第二创新点，不运行24→1、可学习滤波器、动态门控、峰值损失或VMD；后续需重新定义第二创新点并获得用户确认。

## 输出

`experiments/results/causal_frequency_ablation/p1_alternative_selection/168h_6h_<evidence_label>/`

## 验证

```bash
uv run python -m py_compile *.py
uv run python -m unittest -v test_frequency_residual_adapter.py
uv run python run_causal_frequency_p1a.py --quick --device cuda
uv run python run_causal_frequency_p1a.py --device cuda
uv run python summarize_causal_frequency_p1a.py
git diff --check
```

## 实际判定

- 对锁定ST逐种子RMSE改善：0.6749%、0.0433%、0.0815%，平均0.2666%；
- 平均RMSE：局部滤波器50.1954 μg/m³，时域控制49.9519 μg/m³；
- 3/3种子和6/6预测步均未超过时域控制；
- 29项联合测试、严格重构、冻结state hash、禁用退化、容量匹配、源码哈希及独立预测复算通过。

因此失败原因是未满足实际收益与频率特异性门，而不是实现错误。后续不得执行本任务包已列明的禁止阶段。

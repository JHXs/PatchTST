# P0 任务包：无泄漏数据过渡与锁定 ST 基线

## 任务目标

实现独立于旧 `tsai/data/*.npz` 资产的无泄漏数据管道，并在新协议上重建 `degraded_patchtst_clean` 与 `locked_st_clean`。本任务只解决协议过渡，不实现任何频域模块。

## 前置条件

- 以 [`../experiment-protocol.md`](../experiment-protocol.md) 的 FREQ-PROTOCOL-v1 为唯一数据协议。
- 不修改原始站点 CSV，不覆盖通用流程的 `X.npz`、`y.npz`、pipeline 或 `model_params.npz`。
- 锁定 ST 结构为 Top-5、站点身份偏置、预测端空间残差、退化初始化和冻结 PatchTST 主干。

## 允许修改

- 新建频域专用数据、运行器、汇总脚本与测试文件；建议文件名前缀为 `causal_frequency_` 或 `run_causal_frequency_`。
- `plan/frequency/`、`docs/频域分支/` 和 `experiments/results/causal_frequency_ablation/p0_bridge/`。
- 为复用而对 ST 公共函数做最小兼容性修改时，必须证明既有 ST 输出不变。

## 禁止事项

- 不改写 `experiments/results/st_patchtst_ablation/`。
- 不查看 P0 测试指标后调整相关阈值、Top-k、中心站或时间边界。
- 不实现 rFFT、VMD、动态频带或峰值损失。

## 必需产物

1. 目标起点分割、因果缺失处理、训练期站点筛选和训练期标准化代码。
2. train/valid/test 目标 timestamp 两两无交集的自动断言和单元测试。
3. 数据元数据：边界、站点、均值/标准差、q90、样本数及 timestamp 哈希。
4. 2052–2054 三个选择种子的 P0 验证结果和配对摘要。
5. `stage_status.json`，明确 F1 通过或失败及原因。

## 验收与拒绝条件

- 形状、时间边界、标准化和冻结检查全部通过。
- 168→6 的 `locked_st_clean` 相对 `degraded_patchtst_clean` 在三个验证种子上方向一致，平均 RMSE 降幅至少 0.5%。
- 如未通过，停在 F1，先诊断数据过渡；不得通过频域模块掩盖 ST 过渡失败。

## 最小验证

```bash
uv run python -m py_compile *.py
uv run python -m unittest -v test_st_patchtst.py
# 实现后补充频域数据测试的精确命令
git diff --check
```

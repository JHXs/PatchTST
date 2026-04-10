# ST/PatchTST 评估指标改造计划

## Summary

当前 `PatchTST.py` 和 `ST_PatchTST_model.py` 的评估函数都对 `y_true`、`y_pred` 直接做 `flatten()` 后再计算 `mse / rmse / mae`，因此结果会把样本维、指标维、预测步长维全部混合，只得到一个总体汇总值。

本次改造目标是同时保留两类结果：

- 总体指标：保持现有口径，继续对全部元素汇总计算。
- 单指标指标：按每个输出变量分别统计，预测步长维继续汇总。

## Key Changes

1. 重构 `PatchTST.py` 和 `ST_PatchTST_model.py` 的评估逻辑。
   - 保留 `valid`、`test` 两个 split 的总体 `mse / rmse / mae`。
   - 新增按指标拆分的 `mse / rmse / mae`。
   - 逐指标计算时，仅对当前指标自己的 `samples × horizon` 范围做展平。

2. 统一评估结果结构。
   - 评估函数返回一个字典：
     - `summary`：总体指标表，索引为 `valid`、`test`
     - `per_metric`：逐指标明细表，包含 `split, metric_name, mse, rmse, mae`
   - 运行评估时同时打印总体指标和逐指标指标。

3. 为逐指标结果保存可读名称。
   - `data_preparation.py` 在 `model_params.npz` 中保存多站点中心站目标变量名。
   - `data_preparation_single.py` 在 `model_params.npz` 中保存单站点目标变量名。
   - 若历史产物中没有变量名，则评估阶段自动回退为 `metric_0`, `metric_1`, ...。

4. 调整主程序输出。
   - `main.py` 接收新的评估返回结构。
   - 主程序不再假设评估结果是单个 `DataFrame`。

## Test Plan

1. 运行 `python -m py_compile *.py`，确认所有顶层模块语法通过。
2. 重新运行评估流程，确认：
   - 预测张量形状仍为 `[samples, feat_size, horizon]`
   - 总体指标仍能输出 `valid`、`test` 两行
   - 逐指标结果行数为 `2 × feat_size`
   - 每个指标都能看到独立的 `mse / rmse / mae`
3. 抽查单个指标，手工验证 `y[:, i, :]` 和 `pred[:, i, :]` 的计算结果与表中一致。
4. 若重新跑数据准备，确认打印出来的 `metric_name` 使用真实指标名而不是默认占位名。

## Assumptions

- 本次粒度为“按指标拆分”，不细分到每个预测步长。
- 总体指标必须继续保留，并保持和旧版本相同的统计口径。
- 默认存储方式是函数返回 `DataFrame` 并直接打印，不额外输出 CSV。

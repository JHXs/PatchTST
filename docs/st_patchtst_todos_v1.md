# ST-PatchTST 改造 TODOs v1

## 改造目标

- 先完成最小闭环: `多站点输入 + 中心站输出 + 双分支 ST_PatchTST + 正确评估链路`。
- 不改样本切分、不改预测窗口、不改标签定义。
- 先让新 ST 模型能和单站点 PatchTST 按同口径稳定对比。

## 核心 TODO

### 1. `ST_PatchTST_model.py`

- [ ] 删除当前早期空间融合 `Conv2d(num_stations -> 64 -> 1)` 逻辑。
- [ ] 在 `ST_PatchTST.__init__` 增加并保存以下参数:
  - `center_station_idx`
  - `neighbor_hidden_dim`
  - `neighbor_dropout`
- [ ] 保持模型外部接口不变:
  - 输入仍为 `[B, num_stations * feat_size, seq_len]`
  - 输出仍为 `[B, feat_size, pred_len]`
- [ ] 在 `forward` 里先把输入重排为 `[B, num_stations, feat_size, seq_len]`。
- [ ] 明确切分:
  - `center_x`: `[B, feat_size, seq_len]`
  - `neighbor_x`: `[B, num_neighbors, feat_size, seq_len]`
- [ ] 新增邻站辅助分支:
  - 对每个邻站先做特征投影 `Linear` 或 `1x1 Conv`
  - 基于 `center_x` 生成邻站门控权重
  - 对邻站做加权求和，得到 `neighbor_context`
  - 输出形状必须回到 `[B, feat_size, seq_len]`
- [ ] 新增可学习残差系数 `alpha`，初值设小，例如 `0.1`。
- [ ] 形成主干输入:
  - `enhanced_x = center_x + alpha * neighbor_context`
- [ ] `PatchTST` 主干只接收 `enhanced_x`，并保持 `c_in=feat_size`、`c_out=feat_size`。
- [ ] 增加必要的维度校验:
  - `c_in == num_stations * feat_size`
  - `0 <= center_station_idx < num_stations`
  - `num_neighbors >= 1`
- [ ] 在关键 reshape / index 位置补简短注释，避免后面再次看 shape 时返工。

### 2. `train_st_patchtst(...)` in `ST_PatchTST_model.py`

- [ ] 从 `tsai/data/model_params.npz` 读取 `center_station_idx`。
- [ ] 把新参数补进 `arch_config`:
  - `center_station_idx`
  - `num_stations`
  - `feat_size`
  - `neighbor_hidden_dim`
  - `neighbor_dropout`
- [ ] 保留现有训练入口和 `TSForecaster` 用法，避免额外改动训练流程。
- [ ] 训练前打印一次核心配置，便于确认中心站索引和站点数量是否对齐。

### 3. `evaluate_st_patchtst(...)` in `ST_PatchTST_model.py`

- [ ] 保留 valid/test 两段评估逻辑。
- [ ] 明确输出 `MSE`、`MAE`，与单站点 `evaluate_patchtst(...)` 口径一致。
- [ ] 统一返回格式:
  - 要么继续返回 `(results_df, y_test_preds)`
  - 要么改成只返回 `results_df`
- [ ] 在修改 `main.py` 前先确定返回格式，避免主程序接口不一致。

### 4. `data_preparation.py`

- [ ] 在 `station_id_related_list` 生成后，显式计算 `center_station_idx`。
- [ ] 增加断言，确保 `center_station_id` 确实在筛出的站点列表中。
- [ ] 在 `model_params.npz` 中额外保存 `center_station_idx`。
- [ ] 保持以下内容不变:
  - `X` 仍使用全部站点全部变量
  - `y` 仍只使用中心站变量
  - `fcst_history` / `fcst_horizon` 不变
  - `splits` 不变

### 5. `main.py`

- [ ] 改为从 `ST_PatchTST_model.py` 同时导入 `evaluate_st_patchtst`。
- [ ] 训练 ST 模型后，调用 `evaluate_st_patchtst(...)`，不要再调用 `evaluate_patchtst(...)`。
- [ ] 根据 `evaluate_st_patchtst(...)` 的返回值调整接收方式。
- [ ] 打印最终 `valid/test` 指标，确保 ST 与单站点基线按同一格式输出。

## 验证 TODO

### 1. 静态检查

- [ ] 运行 `/opt/miniforge/bin/conda run -n aqi-pre python -m py_compile *.py`

### 2. 前向传播检查

- [ ] 用随机张量做一次最小前向检查:
  - batch size = 1
  - batch size = 正常训练 batch
- [ ] 确认以下 shape 全部正确:
  - `center_x`
  - `neighbor_x`
  - `neighbor_context`
  - `enhanced_x`
  - 最终预测输出

### 3. 训练链路检查

- [ ] 先运行 `/opt/miniforge/bin/conda run -n aqi-pre python data_preparation.py`
- [ ] 再运行 `/opt/miniforge/bin/conda run -n aqi-pre python main.py`
- [ ] 确认以下步骤都能走通:
  - `TSForecaster` 实例化
  - `lr_find`
  - `fit_one_cycle`
  - `valid/test` 指标输出

### 4. 行为检查

- [ ] 验证 `alpha` 初值生效，初始训练阶段不会让邻站分支压过主干。
- [ ] 预留一个 `alpha = 0` 的退化检查，确认主干路径没有被新代码破坏。
- [ ] 若主版本跑通，再考虑补一个“邻站简单平均聚合”的消融版本。

## 额外风险与建议

### 1. `data_preparation_single.py` 的参数文件覆盖风险

- [ ] 当前单站点脚本也会写 `tsai/data/model_params.npz`，而且只保存:
  - `fcst_history`
  - `fcst_horizon`
- [ ] 如果先跑单站点脚本，再跑 `train_st_patchtst(...)`，ST 训练读取参数时可能缺失:
  - `num_stations`
  - `feat_size`
  - `center_station_idx`
- [ ] 这个问题不属于本轮方案核心，但建议顺手处理，二选一即可:
  - 单站点 / 多站点使用不同的 `model_params` 文件名
  - 保持同名文件，但统一保存一套兼容字段

## 建议编码顺序

1. 先改 `data_preparation.py`，把 `center_station_idx` 存下来。
2. 再改 `ST_PatchTST_model.py`，完成双分支结构。
3. 再改 `main.py`，接上正确评估函数。
4. 最后跑 `py_compile` 和最小训练验证链路。

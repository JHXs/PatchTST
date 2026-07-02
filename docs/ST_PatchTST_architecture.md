# ST-PatchTST 模型说明

配套图形文件：[`ST_PatchTST_architecture.drawio`](./ST_PatchTST_architecture.drawio)

## 1. 模型定位

`ST_PatchTST` 定义在 [`ST_PatchTST_model.py`](../ST_PatchTST_model.py)，用于多站点空气质量/气象时间序列预测。它在 PatchTST 的时间建模主干前加入一个轻量的空间邻站增强分支：

- **中心站序列**作为主信号，直接保留并送入预测主干。
- **邻站序列**只作为辅助信息，通过门控聚合生成邻站上下文。
- 最终使用残差形式 `x_enh = x_c + alpha * neighbor_context` 增强中心站表示，再交给 PatchTST 预测未来序列。

## 2. 输入输出

| 名称 | 张量形状 | 含义 |
|---|---:|---|
| 输入 `x` | `[B, S*F, L]` | 批量大小 `B`，站点数 `S`，每站特征数 `F`，历史长度 `L` |
| 重排后 | `[B, S, F, L]` | 将通道维还原为站点维和特征维 |
| 中心站 `x_c` | `[B, F, L]` | `center_station_idx` 指定的目标站点历史序列 |
| 邻站 `x_n` | `[B, S-1, F, L]` | 除中心站外的其他站点序列 |
| 输出 `ŷ` | `[B, F, pred_len]` | 中心站未来 `pred_len` 步的多变量预测 |

## 3. 结构流程

1. **输入重排**  
   `_reshape_input` 将 `[B, S*F, L]` 变为 `[B, S, F, L]`，并检查通道数是否等于 `num_stations * feat_size`。

2. **中心站/邻站拆分**  
   `_split_center_and_neighbors` 根据 `center_station_idx` 得到中心站 `center_x`，同时用 `neighbor_indices` 取出全部邻站。

3. **邻站投影**  
   邻站张量先重排为 `[B*(S-1), F, L]`，经过共享的 `1x1 Conv -> GELU -> Dropout`，投影到隐藏维 `H=neighbor_hidden_dim`，再恢复为 `[B, S-1, H, L]`。

4. **中心站门控**  
   中心站 `center_x` 经过 `AdaptiveAvgPool1d(1) -> Flatten -> Linear(F,H) -> GELU -> Dropout -> Linear(H,S-1)` 得到每个邻站的 gate logits，再通过 `softmax` 变成邻站权重。

5. **邻站上下文聚合**  
   使用 gate 权重对所有邻站隐藏表示加权求和，得到 `[B, H, L]`，再通过 `1x1 Conv(H,F)` 转回 `[B, F, L]`。

6. **残差增强**  
   使用可学习系数 `alpha` 控制邻站信息强度：

   ```text
   alpha = alpha_max * sigmoid(alpha_logit)
   x_enh = center_x + alpha * neighbor_context
   ```

   当 `alpha_max=0` 时模型退化为只使用中心站输入的 PatchTST。

7. **PatchTST 时间预测**  
   增强后的中心站表示 `[B, F, L]` 输入 PatchTST 主干，完成 patch 切分、patch embedding、Transformer Encoder 和预测头输出。

## 4. 当前训练配置

`train_st_patchtst` 从 `tsai/data/model_params.npz` 读取多站点数据参数，并通过 `TSForecaster` 实例化模型。

关键配置：

| 参数 | 当前值 |
|---|---:|
| `batch_size` | `16` |
| `n_layers` | `3` |
| `n_heads` | `4` |
| `d_model` | `16` |
| `d_ff` | `128` |
| `patch_len` | `4` |
| `stride` | `2` |
| `dropout` | `0.2` |
| `neighbor_hidden_dim` | `32` |
| `neighbor_dropout` | `0.1` |
| `alpha_max` | `0.5` |
| `alpha_init` | `0.0` |
| `metrics` | `rmse`, `mse`, `mae` |
| 导出文件 | `tsai/models/ST_PatchTST.pt` |

## 5. 运行方式

先生成多站点训练资产：

```bash
/home/hansel/miniforge3/condabin/conda run -n aqi-pre python data_preparation.py
```

训练和评估时，将 `main.py` 中的：

```python
train_model = "informer"
```

改为：

```python
train_model = "st"
```

然后运行：

```bash
/home/hansel/miniforge3/condabin/conda run -n aqi-pre python main.py
```

## 6. 设计特点

- **保留中心站主信号**：中心站直接走主干，避免邻站噪声完全主导预测。
- **邻站影响可控**：`alpha` 有上限 `alpha_max`，可学习但不会无限放大。
- **轻量空间建模**：没有引入复杂图神经网络，只用共享投影和门控聚合，参数量较小。
- **兼容 PatchTST**：空间增强后的张量仍是 `[B, F, L]`，可以直接复用 PatchTST 主干。

## 7. 注意事项

- `num_stations` 必须大于 1，否则没有邻站分支。
- `c_in` 必须等于 `num_stations * feat_size`。
- `model_params.npz` 必须来自多站点 `data_preparation.py`，不要与单站点数据准备产物混用。
- 当前训练代码中 early stopping、best checkpoint、gradient clipping 回调处于注释状态，如需稳定训练可重新启用。

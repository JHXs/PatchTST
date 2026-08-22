# PatchTST 项目指南

## 一句话了解项目

这是一个基于 PyTorch、fastai 和 tsai 的逐小时空气质量预测项目，包含 PatchTST、ST-PatchTST、CT-PatchTST 和 Informer。当前主要研究对象是 **ST-PatchTST 的空间邻站融合**。

项目有两条独立流程：

1. `data_preparation*.py` + `main.py`：通用 tsai 训练流程。
2. `run_st_patchtst_ablation.py`：公平、可复现的 ST 消融实验，**当前研究优先使用此流程**。

代码与文档冲突时，以“当前代码和测试 > 实验 CSV/JSON > 实验报告 > 旧文档和 Notebook”为准。

## 快速开始

项目要求 Python 3.12+，统一使用 `uv`。当前锁文件使用 AMD ROCm 7.14；ROCm PyTorch 仍通过 `torch.cuda` API 使用 GPU。

```bash
uv sync --python 3.12
uv run python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

不要使用 `requirements.txt` 安装依赖，它是包含本机路径的旧 Conda 快照。

### 通用 tsai 训练

两个数据准备脚本会覆盖同一批 `tsai/data/` 资产，一次只能选择一个：

```bash
# 多站点，供 ST/CT 使用
uv run python data_preparation.py

# 或单站点，供 PatchTST 基线使用
uv run python data_preparation_single.py
```

修改 `main.py`：

```python
train_model = "pa"  # pa/patchtst、st、ct、informer
```

然后运行：

```bash
uv run python main.py
```

### ST 消融实验（推荐）

```bash
# 两轮、单种子的流程冒烟测试
uv run python run_st_patchtst_ablation.py --quick --device cpu

# 第二轮密集/稀疏正式实验：24h → 1h
uv run python run_st_patchtst_ablation.py \
  --history 24 --horizon 1 --epochs 40 --patience 8 \
  --variants degraded_patchtst,st_pairwise_delta_forecast,st_sparse_delta_forecast \
  --output-dir experiments/results/st_patchtst_ablation/round2_forecast_24h_1h

# 第二轮密集/稀疏正式实验：168h → 6h
uv run python run_st_patchtst_ablation.py \
  --history 168 --horizon 6 --batch-size 512 \
  --epochs 30 --patience 6 \
  --variants degraded_patchtst,st_pairwise_delta_forecast,st_sparse_delta_forecast \
  --output-dir experiments/results/st_patchtst_ablation/round2_forecast_168h_6h

# 站点身份偏置正式实验：24h → 1h 与168h → 6h
uv run python run_st_patchtst_ablation.py \
  --history 24 --horizon 1 --epochs 40 --patience 8 \
  --variants degraded_patchtst,st_station_bias_delta_forecast \
  --output-dir experiments/results/st_patchtst_ablation/round2_station_bias_24h_1h
uv run python run_st_patchtst_ablation.py \
  --history 168 --horizon 6 --batch-size 512 \
  --epochs 30 --patience 6 \
  --variants degraded_patchtst,st_station_bias_delta_forecast \
  --output-dir experiments/results/st_patchtst_ablation/round2_station_bias_168h_6h

# 复算第二轮统一摘要、逐预测步和站点门控统计
uv run python summarize_st_patchtst_round2.py

# 复算验证集Top-k选择
uv run python summarize_topk_selection.py

# 最终Top-5确认：24h → 1h
uv run python run_st_patchtst_ablation.py \
  --history 24 --horizon 1 --epochs 40 --patience 8 \
  --seeds 2047,2048,2049,2050,2051 \
  --variants degraded_patchtst,st_sparse_station_bias_delta_forecast \
  --initialize-from-degraded --freeze-backbone --sparse-neighbor-top-k 5 \
  --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_topk5_24h_1h

# 最终Top-5确认：168h → 6h
uv run python run_st_patchtst_ablation.py \
  --history 168 --horizon 6 --batch-size 512 --epochs 30 --patience 6 \
  --seeds 2047,2048,2049,2050,2051 \
  --variants degraded_patchtst,st_sparse_station_bias_delta_forecast \
  --initialize-from-degraded --freeze-backbone --sparse-neighbor-top-k 5 \
  --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_topk5_168h_6h

# 复算五轮稳定性确认、精确符号检验和严格放行状态
uv run python summarize_stability_confirmation.py

# 从头重跑五轮确认（耗时较长；精确命令见脚本）
bash reproduce_stability_confirmations.sh

# 仅使用训练时间段诊断邻站传播滞后
uv run python analyze_neighbor_lag_predictability.py
```

汇总脚本的默认输入与上述四个显式 `--output-dir` 完全一致；也可通过它的四个 `--*-dir` 参数汇总其他目录。未指定 `--output-dir` 时，运行器仍写入 `experiments/results/st_patchtst_ablation/<history>h_<horizon>h/`。带 `_smoke` 后缀的结果不能用于论文。

### 最小验证

```bash
uv run python -m py_compile *.py
uv run python -m unittest -v test_st_patchtst.py
```

## 关键文件

- `main.py`：通用训练入口，通过 `train_model` 选择模型。
- `data_preparation.py`：多站点准备，默认北京中心站 1013。
- `data_preparation_single.py`：单站点准备，当前为广州站 9022。
- `ST_PatchTST_model.py`：当前 ST 模型及 tsai 训练/评估函数。
- `PatchTST.py`、`Informer_model.py`：PatchTST 与 Informer 实现。
- `CT_PatchTST_model.py`、`CT_PatchTST_fixed.py`：实验性 CT 实现，使用前需单独冒烟测试。
- `evaluation_utils.py`：总体和逐目标 MSE/RMSE/MAE。
- `run_st_patchtst_ablation.py`：正式 ST 消融运行器。
- `test_st_patchtst.py`：ST 形状、门控、梯度、消融独立性和退化不变性测试。
- `docs/实验报告/`、`experiments/results/`：当前结论和原始实验产物。

`PatchTST.ipynb` 是历史广州流程，不是当前权威入口。

## 数据与张量约定

原始数据位于：

- 北京：`tsai/data/stations_data/`，站点 1001–1036；
- 广州：`tsai/data/stations_data_Guangzhou/`，站点 9017–9047、9058。

每站有 8,760 条小时记录和 12 个空气质量/气象特征。北京站 1022 的 PM10、NO2、CO、O3、SO2 全部缺失。

多站点通道必须保持“站点优先”顺序。ST 核心形状：

```text
X: [samples, num_stations * feat_size, history]
y: [samples, feat_size, horizon]       # 中心站目标
```

两个准备脚本会共同覆盖：`X.npz`、`y.npz`、`splits.pkl`、两个 pipeline 和 `model_params.npz`。如果 ST 训练提示参数缺少站点字段，重新运行 `data_preparation.py`。

## 当前 ST-PatchTST

模型将输入 `[B,S*F,L]` 重排为 `[B,S,F,L]`。中心站保留为主路径，邻站经共享投影和门控聚合。当前同时支持输入端和预测端两种受控残差；第二轮推荐使用预测端融合：

```text
base_prediction = PatchTST(center_x)
forecast_residual = alpha * tanh(spatial_forecast_head(neighbor_context))
prediction = base_prediction + forecast_residual
```

当前支持中心站单边门控、中心—邻站成对门控、邻站原值/差值融合、空邻站、Top-k 门控、站点身份偏置和分预测步置信门。预测端修正层零初始化；`alpha_max=alpha_init=0` 或 `forward_components(x, disable_spatial=True)` 时严格退化为不受邻站影响的 PatchTST。`spatial_components(x)` 和 `forward_components(x)` 用于诊断门控及空间残差。

文档中的编码器后空间注意力、VMD 和频域分支尚未实现。

## ST 消融实验与当前结论

运行器默认使用北京中心站 1013，只预测 PM2.5，按时间 70%/10%/20% 划分数据，并在相同主干、种子和训练流程下比较：

- `degraded_patchtst`：关闭空间残差的精度基线；
- `st_center_only`：中心站单边门控；
- `st_pairwise`：成对门控并融合邻站绝对值；
- `st_pairwise_delta`：输入端成对门控并融合“邻站－中心站”差值；
- `st_pairwise_delta_forecast`：在 PatchTST 预测后加入有界空间差值修正；
- `st_sparse_delta_forecast`：预测端修正加 Top-7 稀疏门控；
- `st_station_bias_delta_forecast`：预测端修正加可学习站点身份先验。
- `st_sparse_station_bias_delta_forecast`：显式联合Top-k门控和站点身份先验，用于组合候选；默认k=7，最终确认用k=5；不得替代纯稀疏消融。

第二轮探索结果见 `round2_combined_summary.csv`。最终确认模型显式联合Top-5门控与站点身份偏置，并从退化基线初始化、冻结PatchTST主干。Top-5由2024–2026验证集预先选择；在独立的2047–2051五个新种子中，24→1和168→6的RMSE与MAE均为5/5改善，平均配对RMSE降幅分别为3.1188%和0.8220%。

可以表述为“最终固定结构在北京1013站PM2.5的24→1和168→6任务中稳定优于退化PatchTST（以RMSE为主指标）”。内部严格放行状态为true：两任务均满足5/5方向一致、单侧精确符号检验p=0.03125和预注册实际收益门。不能声称所有指标显著提升、大幅提升或跨城市泛化，因为24→1的SMAPE仍上升且尚未更换城市/中心站。频域分支现在可以作为独立消融进入下一阶段，但尚未实现。

当前实验按既定边界使用完整序列筛选相关站点，存在前视信息。新建无泄漏实验时，应先划分时间，再仅使用训练期筛选站点。

## 开发注意事项

- `data_preparation.py` 和 `main.py` 在导入时即执行，不要把它们当测试工具模块导入。
- 数据准备当前从 `df` 而非已计算的 `df_scaled` 构造窗口；修改时必须同步检查训练、评估和反标准化。
- PatchTST、ST、Informer 评估返回 `{"summary", "per_metric"}`；CT 评估仍返回元组。
- 旧文档可能描述已移除的 Conv2d 融合、`patch_len=24` 或旧 `arch_config`，以代码为准。
- 原始站点 CSV 是受版本控制的源数据，不要改写。
- `.gitignore` 已忽略模型、`*.pt`、`*.npz`、缓存和虚拟环境。
- 编辑前检查 `git status`，不要覆盖其他未提交工作。

编码保持 4 空格缩进、`snake_case` 函数/变量和 `PascalCase` 类；在 reshape、transpose 和索引附近保留形状注释。公平实验必须保持数据、种子、主干、优化器、训练流程和最佳模型规则一致。提交前运行测试与 `git diff --check`。

# 因果多尺度频域分支结果目录

该目录只接收第二创新点的实验产物，与 `../st_patchtst_ablation/` 完全隔离。当前已有P0、P1和唯一替代P1A的正式选择结果；P1与P1A均未通过频率特异性门，后续阶段未进入。

## 目录结构

```text
causal_frequency_ablation/
├── p0_bridge/
├── p1_fixed_selection/
├── p1_alternative_selection/
├── p2_learnable_selection/
├── p3_gating_selection/
├── p4_peak_selection/
├── p5_vmd/
├── p6_internal_confirmation/
└── p6_external_generalization/
```

各阶段下按 `<history>h_<horizon>h_<evidence_label>/` 分任务。输出目录和 `experiment_config.json` 必须包含 `smoke`、`selection`、`internal_confirmation` 或 `external_generalization` 证据标签。

P0 当前入口：

```bash
# 冒烟，不参与放行
uv run python run_causal_frequency_p0.py --history 168 --horizon 6 --quick --device cuda

# 168→6 主验证
uv run python run_causal_frequency_p0.py --history 168 --horizon 6 --device cuda

# 24→1 辅助验证
uv run python run_causal_frequency_p0.py \
  --history 24 --horizon 1 --epochs 40 --patience 8 --batch-size 256 \
  --device cuda

# 独立复算两任务并生成整体 Gate F1
uv run python summarize_causal_frequency_p0.py
```

P1固定rFFT与唯一P1A入口：

```bash
# 固定rFFT与等容量时域控制；F3正式结果已判失败
uv run python run_causal_frequency_p1.py --quick --device cuda
uv run python run_causal_frequency_p1.py --device cuda

# 唯一因果局部滤波器替代；FA1正式结果已判失败
uv run python run_causal_frequency_p1a.py --quick --device cuda
uv run python run_causal_frequency_p1a.py --device cuda

# 独立复算FA1完整性和失败门
uv run python summarize_causal_frequency_p1a.py
```

当前停止边界：不得运行P2–P6目录对应的计划实验；这些空目录名只保留原始输出契约。继续研究须先重新定义第二创新点。

## 每个正式任务的最小文件

- `experiment_config.json`
- `dataset_metadata.json`
- `raw_metrics.csv`
- `summary_metrics.csv`
- `paired_differences.csv`
- `per_horizon_metrics.csv`
- `peak_metrics.csv`
- `stage_status.json`
- `auto_report.md`
- `frequency_diagnostics/`、`training_logs/`、`predictions/`、`checkpoints/`

详细字段、统计和完整性要求见 [`../../../plan/frequency/experiment-protocol.md`](../../../plan/frequency/experiment-protocol.md)。带 `_smoke` 后缀或 `smoke` 标签的结果不得用于论文表格、显著性检验或最终结论。选择集和确认集禁止混合汇总。

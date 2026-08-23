# 因果多尺度频域分支结果目录

该目录只接收第二创新点的实验产物，与 `../st_patchtst_ablation/` 完全隔离。当前尚无频域模型正式结果，本文件仅定义输出契约。

## 目录结构

```text
causal_frequency_ablation/
├── p0_bridge/
├── p1_fixed_selection/
├── p2_learnable_selection/
├── p3_gating_selection/
├── p4_peak_selection/
├── p5_vmd/
├── p6_internal_confirmation/
└── p6_external_generalization/
```

各阶段下按 `<history>h_<horizon>h/` 分任务。输出目录和 `experiment_config.json` 必须包含 `smoke`、`selection`、`internal_confirmation` 或 `external_generalization` 证据标签。

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

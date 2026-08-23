# P1 固定频带与容量控制自动摘要

> SELECTION DATA - validation only; test metrics were not computed.

- 任务：168→6
- 当前 Gate：F3 failed

| 变体 | RMSE (μg/m³) | MAE (μg/m³) | 残差RMS比 |
|---|---:|---:|---:|
| locked_st_clean | 50.3299 ± 0.5707 | 32.9070 ± 0.7131 | 0.000000 |
| st_time_residual_adapter | 49.9519 ± 0.4962 | 32.3506 ± 0.6257 | 0.044565 |
| st_fixed_frequency_residual | 50.1534 ± 0.5769 | 32.5723 ± 0.7164 | 0.031116 |

完整配对差值见 `paired_differences.csv`。F3 只有在固定频带同时超过锁定 ST 和等容量时域控制时才通过。

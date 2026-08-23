# P1 固定频带与容量控制自动摘要

> SMOKE TEST DATA - DO NOT USE AS THESIS EVIDENCE.

- 任务：168→6
- 当前 Gate：F2 passed

| 变体 | RMSE (μg/m³) | MAE (μg/m³) | 残差RMS比 |
|---|---:|---:|---:|
| locked_st_clean | 50.4392 ± 0.0000 | 33.0689 ± 0.0000 | 0.000000 |
| st_time_residual_adapter | 50.1743 ± 0.0000 | 32.6677 ± 0.0000 | 0.024351 |
| st_fixed_frequency_residual | 50.3295 ± 0.0000 | 32.9240 ± 0.0000 | 0.010969 |

完整配对差值见 `paired_differences.csv`。F3 只有在固定频带同时超过锁定 ST 和等容量时域控制时才通过。

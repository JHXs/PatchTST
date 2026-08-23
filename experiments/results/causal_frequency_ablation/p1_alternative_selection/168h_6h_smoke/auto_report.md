# FA1 因果局部滤波器组自动摘要

> SMOKE TEST DATA - DO NOT USE AS THESIS EVIDENCE.

- 任务：168→6
- Gate FA1：in_progress
- 分解：low=MA48, mid=MA12-MA48, high=x-MA12

| 变体 | RMSE (μg/m³) | MAE (μg/m³) |
|---|---:|---:|
| locked_st_clean | 50.4392 ± 0.0000 | 33.0689 ± 0.0000 |
| st_time_residual_adapter | 49.9027 ± 0.0000 | 32.2252 ± 0.0000 |
| st_causal_filterbank_residual | 50.3204 ± 0.0000 | 32.9125 ± 0.0000 |

# FA1 因果局部滤波器组自动摘要

> SELECTION DATA - the sole pre-registered alternative after F3 failure.

- 任务：168→6
- Gate FA1：failed
- 分解：low=MA48, mid=MA12-MA48, high=x-MA12

| 变体 | RMSE (μg/m³) | MAE (μg/m³) |
|---|---:|---:|
| locked_st_clean | 50.3299 ± 0.5707 | 32.9070 ± 0.7131 |
| st_time_residual_adapter | 49.9519 ± 0.4962 | 32.3506 ± 0.6257 |
| st_causal_filterbank_residual | 50.1954 ± 0.5592 | 32.6764 ± 0.6986 |

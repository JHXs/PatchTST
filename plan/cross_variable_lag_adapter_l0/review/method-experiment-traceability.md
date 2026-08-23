# 方法—实验追踪

| 主张/问题 | L0 对照 | 产物 | 允许表述 |
|---|---|---|---|
| 七变量是否含自历史之外的近期线性信息 | R_recent vs R_self | paired_differences.csv | 仅选择区间线性预测关联 |
| 离散滞后是否优于 recent-only | R_lag vs R_recent | paired_differences.csv、task_summary.csv | 仅固定 lag 的 L0 证据 |
| 是否达到进入 L1 的预注册硬门 | 168→6 五折 | gate_status.json | PASS 才允许实现；STOP 即归档 |
| 24→1 是否提供支持 | 24→1 五折 | task_summary.csv | 支持性结果，不改变主门 |
| 是否无泄漏 | split/target-set/fit ledger | boundary_checks.csv、fit_ledger.csv | 仅按 CSV 时间戳的历史回放 |

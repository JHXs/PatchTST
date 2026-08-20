# 表格数据规范

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| 主结果表 | 比较三种模型 | 模型 | RMSE/MAE/SMAPE、mean ± std | `summary_metrics.csv` | 实验运行器 |
| 配对改进表 | 比较相同种子差值 | ST模型×种子 | ST-baseline 指标差 | `paired_differences.csv` | 实验运行器 |
| 空间诊断表 | 判断空间分支是否生效 | 模型×种子 | alpha、残差比、熵、null权重 | `raw_metrics.csv` | 实验运行器 |
| 真实性消融表 | 检查邻站依赖 | 模型×干预 | 正常/置零/打乱 RMSE | `raw_metrics.csv` | 实验运行器 |

# L0 表格数据契约

| 文件 | 行定义 | 指标/字段 | 来源 |
|---|---|---|---|
| fold_horizon_metrics.csv | task×fold×model×horizon | RMSE/MAE/SMAPE、alpha、样本数 | outer_predictions.csv 复算 |
| fold_metrics_all.csv | task×fold×model | RMSE_all/MAE_all/SMAPE_all | outer_predictions.csv 复算 |
| paired_differences.csv | task×fold×对照 | 绝对差、相对改善、方向 | fold_metrics_all.csv |
| task_summary.csv | task×model/对照 | 五折均值、方向数 | fold_metrics_all.csv |
| inner_alpha_scores.csv | task×fold×model×alpha | inner RMSE_all、是否选择 | inner 预测即时汇总 |

# 表格契约

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| 表 1 | 展示三轮 alpha 搜索及 O3 补测 | Round 1–3、O3 check | 六个 alpha、valid MSE | `figures/data/alpha_search_trajectory.csv` | 当前实验记录 |
| 表 2 | 展示最佳模型逐污染物收益 | 六种污染物 | alpha=0 MSE、校准 MSE、相对改善 | `figures/data/pollutant_mse_improvement.csv` | 当前实验记录 |
| 表 3 | 展示随机种子复核 | seed 20260809、20260810 | 六个校准 alpha、valid MSE | `figures/data/alpha_seed_stability.csv` | 当前实验记录 |
| 表 4 | 比较四种 alpha 策略 | alpha=0、统一、逐污染物、可学习 | 3-seed MSE/RMSE/MAE 均值、标准差、95% CI | `tsai/models/alpha_ablation/ablation_results.csv` | 补强实验 |
| 表 5 | 报告训练效率 | 四种 alpha 策略 | 时间、完成 epoch、参数量 | `tsai/models/alpha_ablation/ablation_results.csv` | 补强实验 |
| 表 6 | 比较逐污染物收益 | 六种污染物 × 三种空间策略 | 三 seed MSE 均值、标准差、相对 alpha=0 改善 | `figures/data/alpha_ablation_pollutant_summary.csv` | 补强实验 |
| 表 7 | 严格原生 PatchTST 对照 | PatchTST 与四种 ST 策略 | 三 seed MSE/RMSE/MAE、配对改善、时间、参数量 | `tsai/models/patchtst_strict_baseline/baseline_results.csv` 与 alpha 消融结果 | 严格基线实验 |

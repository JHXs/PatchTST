# Figure Data Manifest

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| Alpha 搜索轨迹与 valid MSE | `figures/data/alpha_search_trajectory.csv` | Real | `tsai/models/alpha_search/search_results.csv` 与 O3 补实验 | `figures/alpha_search/fig1_alpha_search_trajectory.py` | PNG, SVG |
| 最佳模型逐污染物 MSE 改善 | `figures/data/pollutant_mse_improvement.csv` | Real | `tsai/models/alpha_o3_check/search_results.csv` | `figures/alpha_search/fig2_pollutant_improvement.py` | PNG, SVG |
| 双随机种子 alpha 稳定性 | `figures/data/alpha_seed_stability.csv` | Real | 主搜索第三轮与 seed 20260810 复核 | `figures/alpha_search/fig3_seed_stability.py` | PNG, SVG |
| 四种 alpha 策略的三 seed 消融与效率 | `figures/data/alpha_ablation_seed_results.csv`、`alpha_ablation_summary.csv`、`alpha_ablation_pairwise.csv`、`alpha_ablation_pollutant_summary.csv` | Real | `tsai/models/alpha_ablation/ablation_results.csv` | `figures/alpha_search/fig4_alpha_strategy_ablation.py` | PNG, SVG |
| 四种策略最终 alpha 的均值与标准差 | `figures/data/alpha_ablation_selected_alpha.csv` | Real | `tsai/models/alpha_ablation/ablation_results.csv` | `figures/alpha_search/fig5_ablation_alpha_heatmap.py` | PNG, SVG |
| 严格中心站 PatchTST 与空间融合策略 | `figures/data/strict_patchtst_seed_results.csv`、`strict_patchtst_summary.csv`、`strict_patchtst_pairwise.csv`、`strict_patchtst_pollutant_improvement.csv` | Real | `tsai/models/patchtst_strict_baseline/baseline_results.csv` 与 alpha 消融结果 | `figures/alpha_search/fig6_strict_patchtst_comparison.py` | PNG, SVG |

# Baseline v2 table schema

| Table | Purpose | Unit / aggregation | Data source |
|---|---|---|---|
| main_table | All new arms, reused capacity curves, traditional baselines, mandatory prior negatives | common `(L,H,seed)`; deterministic arms by config | formal + archives |
| capacity_curve | Parameter/RMSE points; TST remains per `(L,H)` and non-aligned | arm × config mean | formal + archives |
| per_lead | Lead-wise RMSE and paired difference | run and pooled lead | saved predictions |
| stratified_H/L | Horizon/history stratification | common paired subset | paired detail |
| rank_correlation | Validation/test rank agreement | pooled and within-config Spearman | completed runs |
| dual_criterion | Validation-selected and test-selected upper-bound views | common paired subset | completed runs |
| compliance_self_check | Protocol §8 ten gates | boolean gate | independent audits |

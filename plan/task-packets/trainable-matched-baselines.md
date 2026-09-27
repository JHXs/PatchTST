## Task Packet

- Scope: 北京 20 配置的单站点 GRU/LSTM 同可训练参数预算对照，包括预注册、运行器、测试、正式矩阵、汇总、表图和结果文档。
- Files to read: `docs/研究路线总表.md`, `run_beijing_leakfree_coverage.py`, `run_st_patchtst_ablation.py`, `run_capacity_search.py`, `summarize_capacity_search.py`, 既有容量搜索与 S3 数据。
- Files allowed to edit: 用户指定的新增产物，以及本分支的 `plan/`、`tables/table-schema.md`、`figures/data-manifest.md`。
- Required skills: paper-orchestration, experiment-results-planning, figures-python, statistical-analysis, writing-core, verification.
- Evidence/data inputs: `experiments/results/capacity_search/raw_metrics.csv`, S3 最强单站点选择表，本轮真实预测 NPZ。
- Required artifacts: 协议、运行器、汇总器、测试、原始结果、容量曲线 CSV/PNG/SVG、同预算配对表、合规检查、结果文档。
- Rejection checks: 非单通道输入；test 参与早停/选择；参数量不可复算；身份缺失；预测复算不一致；将 cap128 vs h64 误称为同预算。
- Validation commands: 语法检查；`test_trainable_matched.py`；单配置全 12 臂冒烟；正式 768 身份完整性与预测独立复算；汇总重跑哈希一致；图像视觉检查；`git diff --check`。

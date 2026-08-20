## Task Packet

- Scope: 探索比输入端空间差值更有效的 ST-PatchTST 结构，首选预测端有界空间残差，并分析传播滞后的可用性。
- Files to read: `ST_PatchTST_model.py`, `run_st_patchtst_ablation.py`, 第一轮正式实验 CSV/JSON 和中文实验报告。
- Files allowed to edit: `ST_PatchTST_model.py`, `run_st_patchtst_ablation.py`, `test_st_patchtst.py`, `plan/`, `tables/`, `figures/`, `docs/实验报告/`, `experiments/results/st_patchtst_ablation/`。
- Required skills: paper-orchestration, experiment-results-planning, verification。
- Evidence/data inputs: 北京多站 PM2.5 原始 CSV、第一轮逐种子训练日志和预测结果。
- Required artifacts: 新模型变体、退化一致性测试、验证集筛选记录、3种子正式指标、扰动消融、报告增补。
- Rejection checks: 只依据测试集选择超参数；改变基线训练条件；删除负结果；用单种子结果声称稳定提升；预测端残差初始即破坏中心站基线。
- Validation commands: Python语法检查；单元测试；快速实验；正式实验文件完整性和指标重算；`git diff --check`。

## Task Packet

- Scope: 在不改动既有 ST-PatchTST 实验产物和模型代码的前提下，为第二创新点建立独立的因果多尺度频域研究工作区，修订两份既有频域文档，并锁定后续实现、消融、统计确认和止损协议。
- Files to read: `AGENTS.md`、`ST_PatchTST_model.py`、`run_st_patchtst_ablation.py`、`test_st_patchtst.py`、两份既有频域文档、ST 稳定性 CSV/JSON、根目录 `plan/`、`tables/` 与 `figures/`。
- Files allowed to edit: `docs/频域分支/`、`plan/frequency/`、`tables/frequency/`、`figures/frequency/`、`experiments/results/causal_frequency_ablation/README.md`、`AGENTS.md`、新建 `plan/README.md`，以及根 `plan/`、`tables/`、`figures/` 既有权威文件中仅用于区分创新点的标题和导航链接。不得改动第一创新点的实质协议、数据或结论。
- Required skills: `using-research-writing`、`paper-orchestration`、`experiment-results-planning`、`statistical-analysis`、`peer-review`、`verification`。
- Evidence/data inputs: 最终 Top-5 ST 稳定性确认产物；北京 1013 站训练期只读频谱诊断；当前消融运行器的数据选择、切分、初始化和冻结策略；已核实的频域预测与空气质量预测相关工作。
- Required artifacts: 频域项目概览、实施方案、实验协议、阶段门、方法—实验追溯矩阵、结果表/图数据契约、分阶段任务包、结果目录说明、规范审查、质量审查和能力使用审计。
- Rejection checks: 与 ST 文件混写；修改或覆盖 ST 正式结果；把规划阈值写成真实结果；沿用输入端融合；未设置等容量时域适配器和气象信息控制；站点筛选使用完整序列；多步任务跨划分共享目标时间点；在测试集上选择频带或门控；把随机种子稳定性表述成跨城市泛化；把 `causal` 表述成因果推断。
- Validation commands: 路径和链接扫描；Markdown 关键章节检查；频域目录与 ST 目录隔离检查；既有结果哈希/状态不受影响检查；`git diff --check`；对所有新文档执行规范与质量复核。

## Task Packet

- Scope: 修复第二轮审查问题，重建无混杂稀疏消融，并通过多轮互不重复的新种子确认预测端ST-PatchTST的RMSE初始化稳定性；在2042–2046严格门槛仅差0.00307个百分点后，执行最后一次预注册稀疏度验证集选择与2047–2051确认。
- Files to read: `run_st_patchtst_ablation.py`, `analyze_neighbor_lag_predictability.py`, `summarize_st_patchtst_round2.py`, 第二轮正式/筛选CSV、第二轮中文报告和三条review意见。
- Files allowed to edit: 模型/运行器/诊断与汇总脚本、对应测试、`AGENTS.md`、`plan/`、`tables/`、`figures/`、第二轮报告和`experiments/results/st_patchtst_ablation/`。
- Required skills: paper-orchestration、experiment-results-planning、statistical-analysis、verification。
- Evidence/data inputs: 北京中心站1013的19站PM2.5序列；既有2024–2026验证/探索结果；互不重复的确认种子2027–2051。
- Required artifacts: 审查修复、无站点偏置的稀疏筛选/正式结果、修正的滞后CSV、可参数化汇总入口、新种子确认原始数据、统计摘要、中文报告与审查文件。
- Stability gate: 每次确认的两个任务均要求五个预先指定新种子的ST RMSE全部低于配对退化基线；单侧精确符号检验`p=0.03125`；新种子平均配对RMSE降幅至少为24→1的1.0%和168→6的0.5%。MAE/SMAPE为完整报告的次要指标，不用于替换失败的主门槛。最终组为2047–2051。
- Allowed claim after pass: “在固定结构与五个预先指定的新随机种子确认实验中，ST-PatchTST在24→1和168→6任务上均稳定降低RMSE”。不得省略单城市、单中心站、单污染物和次要指标限制。
- Rejection checks: 稀疏变体混入站点偏置；用增广RMSE作相对增益分母；复现命令与汇总目录不一致；看到结果后修改门槛/种子；把探索性2024–2026数据混作独立确认性p值；主门槛失败后宣称稳定。
- Validation commands: 语法、单元测试、两类脚本重跑、目录/种子/预测形状/NaN/指标复算、精确符号检验与Bootstrap CI、`git diff --check`。

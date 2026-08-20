# 进度记录

## 当前阶段

S3 Experiments：实现、运行并记录 ST-PatchTST 对照实验。

## 状态

- [x] 明确退化基线定义和实验边界
- [x] 建立实验协议与结果数据契约
- [x] 完成模型实现
- [x] 完成烟雾测试（验证后已移入系统回收站，避免误作正式结果）
- [x] 完成正式多种子实验
- [x] 生成中文实验报告
- [x] 完成双重审查与能力使用审计

## 实验结论状态

- 直接邻站绝对值融合未优于退化基线，作为负结果保留。
- 空间梯度版本在24→1任务的3个种子上均降低RMSE，平均降低0.1884%，但MAE基本不变。
- 空间梯度版本在168→6任务平均降低RMSE 0.4558%，但仅2/3个种子改善，仍存在初始化敏感性。
- 当前允许的结论为“初步平均改善且扰动消融显示利用了真实邻站差异”；不允许声称显著或全面稳定优于。

### Capability-use audit

- Required skills: paper-orchestration, experiment-results-planning, verification。
- Skills actually used: 上述三项均已使用；未生成论文图，因此未调用 figures-python。
- Inputs consumed: 仓库模型与数据准备代码、ST问题诊断文档、北京35站原始CSV、用户指定的实验边界。
- Inputs not used and why: 广州数据未使用，因为本轮先固定北京中心站1013；12变量联合目标未使用，因为会稀释PM2.5空间效应。
- Artifacts produced: 改进模型、实验运行器、单元测试、3组正式实验数据、统一摘要、中文报告、协议和追溯文件。
- Verification run: `uv run python -m py_compile *.py`；`uv run python -m unittest -v test_st_patchtst.py`；正式实验文件/种子/预测形状/核心均值重算检查；`git diff --check`。
- Remaining risk: 仅3个种子和一个城市；168→6仍有一个种子退化；本轮按用户要求不处理站点筛选前视边界；尚未验证12/24小时预测。

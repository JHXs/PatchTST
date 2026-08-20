# 进度记录

## 当前阶段

S4 Complete：第二轮 ST-PatchTST 结果已固化并通过交付验证。

## 第二轮状态

- [x] 锁定“预测端空间残差”实验假设
- [x] 完成当前结果与数据可预测性诊断
- [x] 实现预测端残差融合及单元测试
- [x] 完成24→1和168→6快速筛选
- [x] 对通过验证集筛选的方案运行3种子正式实验
- [x] 更新统一摘要与中文实验报告
- [x] 完成第二轮规范/质量审查和能力审计

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

## 第二轮结论状态

- 预测端有界空间残差解决了输入端空间修正容易污染中心站序列的问题。
- 站点身份偏置版本在24→1任务中相对本轮退化 PatchTST 平均降低RMSE 2.7353%，3/3个种子改善；MAE基本不变，SMAPE上升。
- 168→6任务平均降低RMSE 1.0618%，RMSE/MAE/SMAPE均值均改善，但仅2/3个种子改善。
- 同一已训练模型中显式禁用空间后，24→1平均RMSE增加0.6133 μg/m³；全局打乱邻站后增加1.4284 μg/m³，支持空间归因。
- 训练期滞后诊断显示18/18个邻站均以同步0小时输入最好，不支持把1/2/3/6/12小时固定时移作为当前主方案。
- 候选结构最终偏好包含探索性正式测试证据；后续需要新种子、中心站或城市确认，不能包装成完全独立的确认性选择。

### 第二轮 capability-use audit

- Required skills: paper-orchestration、experiment-results-planning、verification。
- Skills actually used: 三项均用于任务拆分、验证集筛选协议、结果契约、报告边界和最终检查；未制作论文图，因此未调用 figures-python。
- Inputs consumed: 第一轮正式指标/预测/训练日志，北京19站原始PM2.5序列，模型与运行器代码，用户指定的实验边界。
- Inputs not used and why: 广州数据及其他中心站未使用，因为本轮先固定北京1013站；VMD/频域分支未实现，因为当前优先稳定空间分支；正传播滞后未进入模型，因为训练期诊断不支持。
- Artifacts produced: 预测端模型与候选变体、7项单元测试、验证筛选数据、4组3种子正式实验、逐预测步/门控/滞后诊断CSV、中文报告和追溯资料。
- Verification run: `uv run python -m py_compile *.py`；`uv run python -m unittest -v test_st_patchtst.py`；摘要和滞后脚本重跑；正式实验目录完整性、种子、NaN、预测形状和指标复算；`git diff --check`。
- Remaining risk: 仅3个种子和一个城市；24→1的MAE/SMAPE未同步改善；168→6仍有一个种子退化；候选最终偏好需要独立数据确认；本轮按用户要求不处理站点筛选前视边界。

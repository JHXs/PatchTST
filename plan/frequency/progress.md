# 频域分支进度

## 当前阶段

S3 Experiments 已按止损规则结束：固定rFFT的F3失败，唯一预注册的12/48小时因果局部滤波器组FA1也失败。当前频域/多尺度第二创新点路线停止，等待重新定义第二创新点。

## 实验门状态

- [x] F0 方案锁定
- [x] F1 无泄漏数据过渡
- [x] F2 频域 MVP 正确性
- [x] F3 固定频带特异性（失败并止损）
- [x] FA1 因果局部滤波器组（失败并止损）
- [ ] F4 受约束可学习频带（未进入）
- [ ] F5 动态门控（未进入）
- [ ] F6 峰值感知（未进入）
- [ ] F7 北京 1013 内部确认（未进入）
- [ ] F8 新中心站/城市外部泛化（未进入）

## 已锁定决策

- 频域文件和 ST 文件物理隔离。
- 频域使用预测端残差，不使用 PatchTST 输入端频域增强。
- 首轮只使用中心站 PM2.5，168→6 为主任务。
- 等容量时域残差是必做对照。
- 可学习频带、动态门控和峰值损失必须逐阶段进入。
- VMD 只作后续对照。
- 无泄漏新协议必须重新建立锁定 ST 过渡基线。
- 北京 1013 为内部确认，外部泛化需新站点或广州数据。

## 当前已知数据协议差异

以当前北京 1013 PM2.5 数据做的只读诊断显示：

- 完整序列阈值 0.85 筛选为 19 站；只用训练时间段后为 18 站；
- 训练期筛选相对完整序列移除 1005、1008，加入 1021；
- 现有按滑窗起点分割的 168→6 任务在 train/valid 和 valid/test 边界各有 5 个重叠目标时间点；
- 新协议需用原始目标时间边界消除上述重叠。

上述只是协议诊断，不是频域模型结果。

## P0/F1 结果

- 168→6：锁定 ST 相对退化基线在2052–2054为3/3验证 RMSE 改善，平均1.2891%，最小0.8031%；
- 24→1：3/3改善，平均5.0230%；
- 两任务只计算验证指标，未计算测试指标；
- 整体放行文件：`experiments/results/causal_frequency_ablation/p0_bridge/p0_gate_status.json`；
- 该结果只证明无泄漏 ST 过渡成立，不是频域分支收益。

## P1/F3 结果

- 固定rFFT相对锁定ST为3/3验证RMSE改善，但平均仅0.3502%，未达到0.5%门槛；
- 等容量时域适配器相对ST平均改善0.7498%；
- 固定rFFT相对时域控制3/3退化，平均落后0.4024%，六个预测步均未胜出；
- 因此不能声称频率分解提供增量价值，原P2–P5均停止。

## P1A/FA1 结果

- 因果局部滤波器组相对锁定ST为3/3验证RMSE改善，但分别只有0.6749%、0.0433%和0.0815%，平均0.2666%，未达到0.5%门槛；
- 其平均RMSE为50.1954 μg/m³，等容量时域控制为49.9519 μg/m³，三个种子和六个预测步均未超过时域控制；
- 三路最大重构误差为2.3842e-7，基础ST state hash不变，禁用残差后逐元素等于锁定ST，可训练参数同为3015；
- 失败来自方法效应不足，不是实现正确性、容量、公平初始化或数据泄漏问题。

## 当前停止边界

不运行24→1、可学习滤波器、动态门控、峰值损失、VMD、内部确认或外部泛化。继续研究前必须重新定义第二创新点并得到用户确认；不得围绕当前验证集继续扫描频带、尺度或结构。

### P1A capability-use audit

- Required skills: `using-research-writing`、`paper-orchestration`、`experiment-results-planning`、`statistical-analysis`、`peer-review`、`verification`。
- Skills actually used: 上述六项均用于唯一替代的预注册、容量控制、失败门判断、负结果审查与独立复算。
- Inputs consumed: P0锁定ST检查点、P1固定rFFT正式控制结果、P1A任务包、FREQ-PROTOCOL-v1、2052–2054验证预测和12/48小时滤波器输出。
- Inputs not used and why: 测试集、24→1、2055–2059、P2–P5均未运行，因为FA1主门失败后协议要求立即止损。
- Artifacts produced: 因果局部滤波器组、P1A运行器、第7项频域专项测试、GPU smoke、正式选择结果、两轮审查和阶段状态。
- Verification run: 29项测试；三路因果性与严格重构；源码和复用控制CSV SHA256；九组预测RMSE独立复算；基础ST state hash、严格禁用、参数量、目标时间交集及逐预测步对照检查；`git diff --check`。
- Remaining risk: 当前结果只是否定所测的两个预注册分解在北京1013站168→6验证集上相对等容量时域适配的增量，不证明所有频域方法普遍无效。

### P1 capability-use audit

- Required skills: `using-research-writing`、`paper-orchestration`、`experiment-results-planning`、`statistical-analysis`、`peer-review`、`verification`。
- Skills actually used: 上述六项均用于容量控制、阶段门、负结果审查、独立复算和止损决策。
- Inputs consumed: P0锁定ST检查点、P1任务包、FREQ-PROTOCOL-v1、2052–2054验证预测、频带mask和逐预测步结果。
- Inputs not used and why: 24→1与测试集未运行，因为168→6主门已失败；P2–P5未进入，避免结果驱动搜索。
- Artifacts produced: 固定频带/时域适配器、6项频域专项测试、P1运行器、GPU smoke、正式容量控制结果及双重审查。
- Verification run: 28项测试；mask分割统一与奇偶窗口重构；源码SHA256；九组预测RMSE复算；基础ST state hash、严格禁用和参数量检查；逐预测步与配对效应复算；`git diff --check`。
- Remaining risk: 单一固定全窗口rFFT可能因缺乏时间定位而受限；FA1是唯一允许的机制性替代，不能被解释为确认实验。

### P0 capability-use audit

- Required skills: `using-research-writing`、`paper-orchestration`、`experiment-results-planning`、`statistical-analysis`、`peer-review`、`verification`。
- Skills actually used: 上述六项均用于阶段约束、P0实现边界、配对效应、双重审查和独立复算。
- Inputs consumed: FREQ-PROTOCOL-v1、P0任务包、旧ST运行器/模型/测试、北京1001–1035原始PM2.5 CSV、2052–2054真实训练日志和预测。
- Inputs not used and why: 测试集指标未计算，因为P0属于验证选择；广州和其他中心站未读取，因为当前不是外部泛化阶段。
- Artifacts produced: 无泄漏数据模块、P0运行器、12项专项测试、两任务selection结果、整体汇总器、Gate状态和双重审查。
- Verification run: `uv run python -m py_compile *.py`；22项ST/P0测试；168→6 GPU smoke；两任务正式运行；预测RMSE、配对效应、时间交集、源码指纹、检查点与主干hash独立复算；`git diff --check`。
- Remaining risk: F1只证明锁定ST在新协议下可用；频率特异性、测试确认和跨站点泛化均未验证。

## Capability-use audit

- Required skills: `using-research-writing`、`paper-orchestration`、`experiment-results-planning`、`statistical-analysis`、`peer-review`、`verification`。
- Skills actually used: 上述六项均用于工作区隔离、任务包、协议、阶段门、统计边界、双重评审与验证。
- Inputs consumed: 两份旧频域方案、ST 模型/运行器/单测、最终 Top-5 确认 CSV/JSON、现有计划与报告、北京 1013 只读数据诊断和已核实频域相关工作。
- Inputs not used and why: 未读取广州或其他中心站作选择，因外部目标必须在后续确认前另行预注册；未生成 mock 结果，因当前不撰写结果章。
- Artifacts produced: 独立频域文档、项目概览、大纲、决策记录、实验协议、阶段门、任务包、表/图契约、结果目录契约和双重评审。
- Verification run: 本地 Markdown 链接 20/20 存在；未发现旧频域文档路径引用或新增文档行尾空白；ST 三个核心代码文件和 `experiments/results/st_patchtst_ablation/` 无差异；`uv run python -m py_compile *.py` 通过；`uv run python -m unittest -v test_st_patchtst.py` 通过 10/10；`git diff --check` 通过；规范审查与质量审查均通过（仅限规划层面）。
- Remaining risk: 频域代码、无泄漏 ST 过渡基线、频率特异性和外部泛化均尚未实验验证；当前文档不构成模型有效性证据。

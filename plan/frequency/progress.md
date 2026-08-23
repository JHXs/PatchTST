# 频域分支进度

## 当前阶段

S3 Experiments / Gate F0 已完成：研究定位、方法方案、实验协议、阶段门和产物契约已建立。Gate F1 及后续实现/实验尚未开始。

## 实验门状态

- [x] F0 方案锁定
- [ ] F1 无泄漏数据过渡
- [ ] F2 频域 MVP 正确性
- [ ] F3 固定频带特异性
- [ ] F4 受约束可学习频带
- [ ] F5 动态门控
- [ ] F6 峰值感知
- [ ] F7 北京 1013 内部确认
- [ ] F8 新中心站/城市外部泛化

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

## 下一个唯一允许任务

执行 [`task-packets/p0-causal-data-bridge.md`](task-packets/p0-causal-data-bridge.md)：实现无泄漏 Dataset/运行器骨架，首先重建 `degraded_patchtst_clean` 与 `locked_st_clean`。在 Gate F1 通过前，不实现可学习频带、气象门控、峰值损失或 VMD。

## Capability-use audit

- Required skills: `using-research-writing`、`paper-orchestration`、`experiment-results-planning`、`statistical-analysis`、`peer-review`、`verification`。
- Skills actually used: 上述六项均用于工作区隔离、任务包、协议、阶段门、统计边界、双重评审与验证。
- Inputs consumed: 两份旧频域方案、ST 模型/运行器/单测、最终 Top-5 确认 CSV/JSON、现有计划与报告、北京 1013 只读数据诊断和已核实频域相关工作。
- Inputs not used and why: 未读取广州或其他中心站作选择，因外部目标必须在后续确认前另行预注册；未生成 mock 结果，因当前不撰写结果章。
- Artifacts produced: 独立频域文档、项目概览、大纲、决策记录、实验协议、阶段门、任务包、表/图契约、结果目录契约和双重评审。
- Verification run: 本地 Markdown 链接 20/20 存在；未发现旧频域文档路径引用或新增文档行尾空白；ST 三个核心代码文件和 `experiments/results/st_patchtst_ablation/` 无差异；`uv run python -m py_compile *.py` 通过；`uv run python -m unittest -v test_st_patchtst.py` 通过 10/10；`git diff --check` 通过；规范审查与质量审查均通过（仅限规划层面）。
- Remaining risk: 频域代码、无泄漏 ST 过渡基线、频率特异性和外部泛化均尚未实验验证；当前文档不构成模型有效性证据。

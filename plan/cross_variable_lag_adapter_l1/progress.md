# L1 进度记录

日期：2026-08-24。

当前阶段：正式L1选择已完成，独立完整性复算PASS，性能阶段门STOP；路线归档，不进入confirmation。

- [x] 确认旧两个 smoke 各有18个PatchTST持久BatchNorm buffer变化，并将旧CPU smoke/汇总判为无效。
- [x] F10每轮train后强制整个PatchTST子模块eval，仅空间分支训练。
- [x] 建立F00 checkpoint→F10初始化→逐epoch→最佳checkpoint→最终加载的独立hash与逐元素参数/buffer断言。
- [x] 新增真实PatchTST BatchNorm一步训练回归测试，并证明空间分支确实更新。
- [x] 汇总器直接加载F00/F10 checkpoint复核`patch_tst.*`逐元素一致，不信任runner声明。
- [x] manifest记录Git commit、dirty/porcelain和六个关键L1源码SHA256；正式入口在读数据/建目录前要求clean tree。
- [x] 从空目录重跑两个CPU全流程smoke，且联合复算完成。
- [x] 新增168→6、F11、真实batch512 GPU资源smoke脚本与独立汇总验收契约。
- [x] 在真实AMD Radeon RX 7600上完成168→6、F11、batch512前向/反向资源smoke。
- [x] 不带`--allow-missing-gpu-smoke`严格重跑联合复算；GPU资源检查与总体integrity均通过。

## 正式目录预登记

- `experiments/results/cross_variable_lag_adapter/l1_selection_24h_1h`
- `experiments/results/cross_variable_lag_adapter/l1_selection_168h_6h`
- `experiments/results/cross_variable_lag_adapter/l1_selection_summary`

三者均不存在。dirty-tree正式入口已实测fail-closed，未读取数据或创建目录。

## 作废与替代 Smoke

旧无效 smoke 已整体移至`/tmp/l1_bn_invalid_smoke_20260824_7Gq22k/`。修复后中间产物因汇总器源码hash随后变化，也主动移至`/tmp/l1_p1_intermediate_smoke_20260824_sqgflt/`，不作为最终证据。

当前预提交工程 smoke：

- `experiments/results/cross_variable_lag_adapter/l1_selection_24h_1h_smoke/`
- `experiments/results/cross_variable_lag_adapter/l1_selection_168h_6h_smoke/`
- `experiments/results/cross_variable_lag_adapter/l1_selection_summary_smoke/`

两个任务均为1 seed、1 epoch、64 train/48 validation和8个完整变体。每个F10四阶段PatchTST持久/完整状态变化数均为0；CPU smoke manifest保留其预提交工程属性。独立复算性能状态为`NOT_EVALUABLE_SMOKE`。GPU资源smoke在干净源码提交`a41b00e`上运行，所有CPU、数据、checkpoint与GPU资源检查均为true，总体integrity=true。

## 验证（2026-08-24）

- `uv run --frozen python -m py_compile *.py`：通过。
- `uv run --frozen python -m unittest -v test_st_patchtst.py test_cross_variable_lag_l0.py test_cross_variable_lag_l1.py`：36/36通过。
- 两个CPU smoke：从空目录重跑通过；关键源码hash与交付源码一致。
- GPU资源命令：在AMD Radeon RX 7600上完成真实batch512前向/反向；峰值分配/保留显存为438,379,520/591,396,864字节。
- `summarize_cross_variable_lag_l1.py --smoke`严格联合复算完成：gate=`NOT_EVALUABLE_SMOKE`，GPU资源检查=true，integrity=true。
- 非smoke入口：dirty tree门按设计退出1，正式目录仍不存在。
- 最终`git diff --check`、未跟踪文本空白检查、无bfill扫描、关键源码manifest hash、F00/F10 checkpoint逐元素对比和源CSV状态检查：通过。

### Capability-use audit

- Required skills：using-research-writing、paper-orchestration、experiment-results-planning、verification。
- Skills actually used：全部使用；分别用于阶段路由、P1补充任务包/双审、资源与产物契约、最终机械验证门。
- Inputs consumed：AGENTS、理论03/04/05、L0源码及权威config/integrity、现有ST模型/运行器/测试、reviewer P1裁决、北京全量时间戳和前7008行允许特征。
- Inputs not used and why：正式test特征/目标/窗口/预测/缓存/指标及确认种子因阶段门STOP未使用；广州数据未进入本轮北京选择。
- Artifacts produced：冻结/追溯修复、GPU资源脚本、15项L1单测（相关总计36项）、逐阶段PatchTST审计、CPU/GPU smoke、两任务三种子正式选择原始产物及统一复算。
- Remaining risk：结论限于北京1013单年度历史回放选择区间；数据血缘与在线报送时刻仍不可证。STOP禁止进一步消费test或确认资源。

## 正式L1结果与最终决定（2026-08-24）

- 24→1和168→6均按冻结预算运行种子2060–2062及全部八个变体；test六项访问计数为0。
- 独立汇总目录：`experiments/results/cross_variable_lag_adapter/l1_selection_summary/`；完整性PASS，正式gate=STOP。
- 168→6 F11相对F10为3/3改善、平均0.477392%，未达到0.5%。
- 对B1/B2/B-flat的平均改善为0.073843%/0.022306%/−0.161018%，对应硬门全部失败。
- M-shuffle只有2/3种子满足增益损失≥50%，种子2061为19.1907%。
- 24→1非劣、高污染安全和逐horizon门通过，但不能挽救主门。
- 最终动作：STOP并归档；不创建确认分支、不打开正式test、不扫描配置或切换备选路线。

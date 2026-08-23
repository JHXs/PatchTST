# L1 进度记录

日期：2026-08-24。

当前阶段：Reviewer P1 工程修复已完成；两个 CPU smoke 已从头替换并独立复算。真实 GPU batch512 资源验收因当前容器未暴露设备而阻断，故 S5 Review 尚未关闭，正式 `2060–2062` 运行未授权且未执行。

- [x] 确认旧两个 smoke 各有18个PatchTST持久BatchNorm buffer变化，并将旧CPU smoke/汇总判为无效。
- [x] F10每轮train后强制整个PatchTST子模块eval，仅空间分支训练。
- [x] 建立F00 checkpoint→F10初始化→逐epoch→最佳checkpoint→最终加载的独立hash与逐元素参数/buffer断言。
- [x] 新增真实PatchTST BatchNorm一步训练回归测试，并证明空间分支确实更新。
- [x] 汇总器直接加载F00/F10 checkpoint复核`patch_tst.*`逐元素一致，不信任runner声明。
- [x] manifest记录Git commit、dirty/porcelain和六个关键L1源码SHA256；正式入口在读数据/建目录前要求clean tree。
- [x] 从空目录重跑两个CPU全流程smoke，且联合复算完成。
- [x] 新增168→6、F11、真实batch512 GPU资源smoke脚本与独立汇总验收契约。
- [ ] 在真实GPU上完成batch512前向/反向资源smoke：当前`torch.cuda.is_available()=false`、设备数0、无`/dev/kfd`与`/dev/dri`。
- [ ] GPU资源smoke成功后，不带`--allow-missing-gpu-smoke`重跑联合复算并关闭S5。

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

两个任务均为1 seed、1 epoch、64 train/48 validation和8个完整变体。每个F10四阶段PatchTST持久/完整状态变化数均为0；manifest均为`git_dirty=true`、`precommit_engineering_smoke=true`。独立复算性能状态为`NOT_EVALUABLE_SMOKE`；所有CPU/数据/checkpoint检查为true，GPU资源检查为false，所以总体integrity按设计为false。

## 验证（2026-08-24）

- `uv run --frozen python -m py_compile *.py`：通过。
- `uv run --frozen python -m unittest -v test_st_patchtst.py test_cross_variable_lag_l0.py test_cross_variable_lag_l1.py`：36/36通过。
- 两个CPU smoke：从空目录重跑通过；关键源码hash与交付源码一致。
- GPU资源命令：真实执行，因无可见GPU按设计退出1；未创建伪GPU产物。
- `summarize_cross_variable_lag_l1.py --smoke --allow-missing-gpu-smoke`：CPU联合复算完成，gate=`NOT_EVALUABLE_SMOKE`，GPU状态=`BLOCKED_NO_EXPOSED_GPU_DEVICE`，integrity=false。
- 非smoke入口：dirty tree门按设计退出1，正式目录仍不存在。
- 最终`git diff --check`、未跟踪文本空白检查、无bfill扫描、关键源码manifest hash、F00/F10 checkpoint逐元素对比和源CSV状态检查：通过。

### Capability-use audit

- Required skills：using-research-writing、paper-orchestration、experiment-results-planning、verification。
- Skills actually used：全部使用；分别用于阶段路由、P1补充任务包/双审、资源与产物契约、最终机械验证门。
- Inputs consumed：AGENTS、理论03/04/05、L0源码及权威config/integrity、现有ST模型/运行器/测试、reviewer P1裁决、北京全量时间戳和前7008行允许特征。
- Inputs not used and why：正式test特征/目标/窗口/预测/缓存/指标因协议禁止未使用；正式三种子因未授权未运行；GPU batch数据因设备未暴露而未进入GPU。
- Artifacts produced：冻结/追溯修复、GPU资源脚本、15项L1单测（相关总计36项）、逐阶段PatchTST审计、两个替代CPU smoke和fail-closed联合复算。
- Remaining risk：真实GPU batch512资源验收未完成；L1正式性能、全部硬门和第二创新点成立性仍未知。

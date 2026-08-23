# L1 质量审查（P1 修复后）

日期：2026-08-24。

## 结论

P1代码修复与替代CPU smoke质量为PASS；整体启动状态为BLOCKED，因为当前环境未完成真实GPU batch512资源验收。旧CPU smoke因PatchTST BatchNorm冻结缺陷全部无效，不得与替代结果并列或合并。

## 修复质量

- 冻结语义从“参数无梯度”提升为“参数无梯度 + 整棵PatchTST eval + 全参数/buffer逐元素身份”。F10每轮train后立即重锁，避免后续重构再次打开BN/dropout。
- 状态证据具有独立来源：F00 checkpoint子state、F10初始化现场、逐epoch现场、重新读取的最佳checkpoint和最终加载现场分别计算。`base_state_before_sha256`与`base_state_after_sha256`也改为真实训练前后状态，不再重复末态hash。
- 回归测试不是dummy：先验证真实PatchTST含BatchNorm，再显式检查权重、running mean/var、计数器，并同时验证空间分支更新，防止“把整个F10都eval/冻结”这种伪修复。
- 汇总器直接读取checkpoint复核F00→F10 lineage，并核对逐epoch审计条数；runner与summarizer不能仅靠同一个布尔字段互相背书。
- manifest记录HEAD、dirty明细和关键源码SHA256。替代smoke是审查前的dirty工程smoke，标识清楚；正式入口对dirty tree fail-closed。

## 替代结果边界

- 两个CPU smoke均从空目录重跑，F10四阶段持久与完整状态变化数为0；联合复算CPU部分通过，性能gate仍为`NOT_EVALUABLE_SMOKE`。
- 独立GPU资源脚本要求真实168→6/F11/512样本完整batch，记录峰值显存且不解释性能；禁止CPU回退。
- 当前容器没有`/dev/kfd`/`/dev/dri`，PyTorch报告0个GPU。脚本真实执行并中止，没有制造成功JSON。联合复算因此保持`integrity=false`和`BLOCKED_NO_EXPOSED_GPU_DEVICE`。

## 剩余风险与启动条件

- 在暴露真实ROCm/CUDA设备的环境完成GPU batch512 smoke，并用严格模式重新联合复算，是正式运行前硬阻断。
- 审查通过后由用户提交代码，使formal clean-tree门可满足；本轮不commit/push。
- L1正式性能、硬门和第二创新点成立性仍完全未知。不得根据任何smoke数值调节结构、lag、beta、门槛或训练预算，也不得自行运行正式2060–2062。

# L1 质量审查（P1 修复后）

日期：2026-08-24。

## 结论

P1代码修复、替代CPU smoke、真实GPU batch512资源验收与严格联合复算均为PASS。旧CPU smoke因PatchTST BatchNorm冻结缺陷全部无效，不得与替代结果并列或合并。

## 修复质量

- 冻结语义从“参数无梯度”提升为“参数无梯度 + 整棵PatchTST eval + 全参数/buffer逐元素身份”。F10每轮train后立即重锁，避免后续重构再次打开BN/dropout。
- 状态证据具有独立来源：F00 checkpoint子state、F10初始化现场、逐epoch现场、重新读取的最佳checkpoint和最终加载现场分别计算。`base_state_before_sha256`与`base_state_after_sha256`也改为真实训练前后状态，不再重复末态hash。
- 回归测试不是dummy：先验证真实PatchTST含BatchNorm，再显式检查权重、running mean/var、计数器，并同时验证空间分支更新，防止“把整个F10都eval/冻结”这种伪修复。
- 汇总器直接读取checkpoint复核F00→F10 lineage，并核对逐epoch审计条数；runner与summarizer不能仅靠同一个布尔字段互相背书。
- manifest记录HEAD、dirty明细和关键源码SHA256。替代smoke是审查前的dirty工程smoke，标识清楚；正式入口对dirty tree fail-closed。

## 替代结果边界

- 两个CPU smoke均从空目录重跑，F10四阶段持久与完整状态变化数为0；联合复算CPU部分通过，性能gate仍为`NOT_EVALUABLE_SMOKE`。
- 独立GPU资源脚本要求真实168→6/F11/512样本完整batch，记录峰值显存且不解释性能；禁止CPU回退。
- Herdr内的受限代理容器未暴露GPU时脚本按设计中止；随后在主机真实ROCm环境使用AMD Radeon RX 7600完成batch512前向/反向，峰值分配/保留显存约418/564 MiB，基础与PatchTST状态前后完全一致。严格联合复算确认GPU资源项及总体integrity均为true。

## 正式选择质量结论

- 两个任务、三个种子和八个变体已在clean tree上按冻结预算执行；每个manifest记录实际commit与源码哈希。
- 独立汇总从validation逐预测重建指标、2×2、场景、M-shuffle与硬门；完整性PASS，test六项访问计数均为0。
- STOP不是单一0.5%门的边界判定：F11同时未击败B1/B2，并在3/3种子上落后等输入B-flat；种子2061的M-shuffle也失败。因此当前结构缺少预注册的特异性证据。
- 24→1、高污染和逐horizon结果如实保留，但没有被用于替代主门。

最终质量裁决：**接受STOP并归档**。不得调节结构、lag、beta、门槛或训练预算后复用当前选择资源，也不得进入正式test或confirmation。

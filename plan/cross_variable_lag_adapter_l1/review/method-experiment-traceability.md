# L1 方法—实验追踪

| Contribution/risk | Method module | Experiment/check | Artifact | Allowed claim | Status |
|---|---|---|---|---|---|
| 增量信息 | M + 冻结 F10 | F11 对 F10/B1 | prediction/paired gate | 仅正式三种子过门后可称增量价值 | 待正式运行 |
| lag 特异性 | 6 个固定 lag | M 对 B2 | paired gate | 仅过门后可称 lag 结构有用 | 待正式运行 |
| 结构特异性 | token gate/group sum | M 对同输入 B-flat | parameter audit/paired gate | 仅过门后可称结构价值 | 待正式运行 |
| 条件门 | q_h 全局 softmax | M 对 B3 | gate diagnostics | 仅称条件门消融 | 待正式运行 |
| 空间×辅助组合 | F00/F01/F10/F11 | 2×2 复算 | factorial_metrics.csv | 预测交互，不作因果解释 | 待正式运行 |
| 真实辅助依赖 | 完整块 M-shuffle | 每训练种子五置换 | shuffle_metrics.csv | 每种子过50%后才称依赖真实对齐 | 待正式运行 |
| 安全与冻结 | 零输出层/有界旁路；F10每轮PatchTST eval | 真实BN一步回归、逐epoch/最佳/最终逐元素状态、汇总器直读checkpoint | unit tests/checkpoint map/patchtst_state_audit | CPU工程smoke可称P1冻结实现已修复；不含性能 | PASS（CPU工程） |
| 资源可运行性 | 168→6 F11，真实batch512，冻结F10 | 单batch GPU前向/反向、峰值显存、基础状态 | `l1_168h_6h_batch512_gpu_smoke/` | 只允许报告资源事实，不解释性能 | PASS（RX 7600，严格复算通过） |
| 无泄漏 | 前80%特征、train-only fit | 访问计数/target sets/ledger | metadata/ledger/tests | 可称选择管线未构造 test 数据 | PASS（工程） |

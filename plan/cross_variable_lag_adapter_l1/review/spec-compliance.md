# L1 规范符合性审查（P1 修复后）

日期：2026-08-24。范围：冻结实现、测试、替代CPU smoke、GPU资源入口和独立复算；不审查正式性能。

| 冻结要求 | 修复后证据 | 状态 |
|---|---|---|
| test隔离 | 只全量读time，特征止于7008；六项test构造/访问计数为0 | PASS |
| 70/10/20与目标集合 | 6132/876/1752；train/validation/reserved-test timestamps两两空交 | PASS |
| pool/训练期筛站/P=7 | 1001–1036含1036、排除1013；raw train筛站；复用L0清洗且无bfill | PASS |
| F10训练态 | 每轮`model.train()`后强制`patch_tst.eval()`；真实BN回归测试覆盖 | PASS |
| F00→F10完整状态 | 来源checkpoint与初始化现场独立hash；逐epoch、最佳checkpoint、最终加载逐元素参数/buffer断言 | PASS |
| 独立checkpoint复核 | 汇总器直接加载F00/F10 checkpoint并逐键逐元素比较`patch_tst.*` | PASS |
| 空间分支可训练 | 真实BN回归测试同时要求至少一个非PatchTST空间参数更新 | PASS |
| 共享冻结基础 | F01共享F00；F11/B1/B2/B-flat/B3共享F10且base eval/no-grad/hash不变 | PASS |
| M/对照/参数门 | P=7,K=6,d=16；1532/6547；B-flat宽15/47且≤3%；结构/超参未改 | PASS |
| 梯度/旁路/组质量 | 首步/后续梯度、精确旁路、全局softmax组和均有断言 | PASS |
| 2×2/L1门/M-shuffle | 独立复算公式、逐horizon/场景和固定块置换契约未改 | PASS（流程） |
| smoke替换 | 旧两任务各18个BN buffer变化的产物已作废；两个CPU smoke从空目录替换 | PASS |
| Git追溯 | manifest含commit、dirty/porcelain与关键源码SHA256；正式要求clean tree | PASS |
| 168→6 GPU batch512 | AMD Radeon RX 7600完成真实F11最大结构/冻结base前向反向；峰值分配/保留显存438,379,520/591,396,864字节；严格复算通过 | PASS |
| 正式执行 | 24→1绑定`5d2e3a1`、168→6绑定`372ff8a`；关键源码哈希一致；2060–2062全部运行 | PASS |
| test边界 | 六项test访问计数为0；只使用validation形成正式选择门 | PASS |
| 正式阶段门 | 独立完整性PASS；F10/B1/B2/B-flat/M-shuffle硬门失败，gate=STOP | PASS（规则执行） |

结论：实现、资源与正式执行均符合冻结协议。正式L1阶段门为 **STOP**，因此规范要求的后续动作是归档路线并禁止confirmation/test/配置回调。

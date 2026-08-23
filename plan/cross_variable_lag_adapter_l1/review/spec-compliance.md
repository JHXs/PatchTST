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
| 168→6 GPU batch512 | 脚本已固定真实F11最大结构/冻结base/显存记录，但当前容器无GPU设备 | BLOCKED |
| 正式边界 | dirty正式入口fail-closed；三个正式目录不存在；未运行三种子、未commit/push | PASS |

结论：Reviewer P1 的冻结与追溯缺陷已修复，替代CPU smoke符合工程流程；旧 smoke 明确无效。由于真实 GPU batch512 资源 smoke 尚未完成，整体裁决为 **BLOCKED / NOT READY FOR FORMAL RUN**，不是L1性能结论。

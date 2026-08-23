# MC-CVLRA L1 实现与 Smoke 审计报告

日期：**2026-08-24**。分支：`experiment/cross-variable-lag-adapter-ablation`。起点提交：`c3da238`。

> 本报告只审计冻结 L1 的实现、测试和工程 smoke。没有执行 `2060–2062` 三种子正式选择，没有读取或构造正式 test 特征、目标、样本、预测、缓存或指标，也不解释 smoke 性能。MC-CVLRA 的 L1 性能状态仍为“未运行”。

## 1. Reviewer P1 与旧 smoke 作废

独立 reviewer 发现首轮 F10 训练在调用整个 `model.train()` 后，只依赖 `requires_grad=False` 冻结 PatchTST 参数，没有把 PatchTST 子模块切回 eval。结果是两个旧 CPU smoke 的 PatchTST 各有 18 个持久 BatchNorm buffer 变化，因而不满足“冻结主干”语义。旧 `l1_selection_24h_1h_smoke`、`l1_selection_168h_6h_smoke` 与联合汇总全部判为无效，不再作为任何实现验收证据；旧目录已整体留档至 `/tmp/l1_bn_invalid_smoke_20260824_7Gq22k/`，没有与替代产物混写。

修复后的 F10 每轮先进入 train，再立即对 `model.patch_tst` 强制 eval；PatchTST 全部参数保持 `requires_grad=False`，空间分支保留 train。运行器从 F00 最佳 checkpoint 独立提取 PatchTST 子 state，再从完成 F00 初始化的 F10 现场独立计算 hash，并逐元素比较。每个 epoch 的训练批后和验证后、每次最佳 checkpoint 写盘并重新读取后、最终最佳 checkpoint 加载后，均断言 PatchTST 全部参数和 buffers 不变。`patchtst_state_audit.csv` 保存各阶段重新计算的持久 state hash、完整参数/buffer hash、张量数、差异张量名与相等状态，不以重复填写同一个训练后 hash 代替审计。

## 2. 回归测试与追溯修复

新增回归测试使用真实 ST-PatchTST/PatchTST，而非 dummy 网络，并先断言其中确实存在 BatchNorm。一次 F10 优化 step 后，测试逐元素核对 PatchTST 全部权重与 buffers，并显式核对每层 BatchNorm 的 `weight`、`running_mean`、`running_var`、`num_batches_tracked` 均等于 F00 初始化现场；同时要求至少一个空间分支参数实际更新。另有运行时断言覆盖逐 epoch、最佳 checkpoint 和最终加载。

独立汇总器不信任 CSV 中的布尔声明：它直接加载同种子 F00 与 F10 checkpoint，提取 `patch_tst.*` 子 state 并逐键、逐元素比较；同时核对 F00 checkpoint、F10 初始化、F10 checkpoint、F10 最终加载四类独立 hash，以及逐 epoch 审计行数与训练日志行数一致。

每个 run manifest 新增完整 Git commit、dirty 状态、porcelain 明细和六个关键 L1 源码 SHA256。正式运行在加载数据或创建正式目录前要求 clean tree；本轮未提交代码，因此替代 smoke 均明确记录为 `git_dirty=true`、`precommit_engineering_smoke=true` 和“PRE-COMMIT DIRTY-TREE ENGINEERING SMOKE”。实际尝试非 smoke 入口已被 clean-tree 门中止，正式目录没有创建。

## 3. 数据、模型和协议边界

无泄漏数据管线仍固定北京1013、原始时间戳精确6132/876/1752的70/10/20边界，只加载前7008行 train+validation 特征。test特征、目标、样本、预测、缓存和指标六项计数均为0，三组目标时间集合两两无交。站点池仍为1001–1036、含1036并排除1013；完整性和Pearson只使用原始训练片。P=7清洗、变量顺序、pressure换算、零值缺失、最多6 h因果ffill和历史回放规则未改，无bfill。

F00/F01/F10/F11、B1、B2、B-flat、B3的结构、P/K/lag、beta、优化器、batch、早停、门槛与正式目录均未改。F01共享F00；F11/B1/B2/B-flat/B3共享F10并在适配器训练期间冻结整个基础模型为 eval/no-grad。M参数数仍为1532/6547，B-flat宽度仍为15/47、参数数1532/6540。

## 4. 替代 CPU smoke 与独立复算

修复冻结和追溯代码后，从空目录重新执行24→1与168→6两个 CPU 全流程 smoke。每个任务仍只使用种子2060、1 epoch、64个训练起点、48个验证起点和全部8个变体；这只是预提交工程 smoke，不消费正式三种子选择。

两份替代产物的 F10 均包含“F00 checkpoint 初始化现场、epoch 后、最佳 checkpoint、最终加载”四阶段记录；持久 state 与完整参数/buffer变化数均为0，四阶段持久 hash 各自重算后只有一个唯一值。独立联合复算直接验证 checkpoint 文件hash、F00→F10 PatchTST逐元素血缘、逐epoch状态、基础eval/no-grad、梯度契约、组质量、参数数、站点筛选、目标时间集合、2×2、逐horizon、场景和M-shuffle。smoke性能 gate固定为`NOT_EVALUABLE_SMOKE`，不展示或使用相对优劣。

替代产物：

- 24→1：`experiments/results/cross_variable_lag_adapter/l1_selection_24h_1h_smoke/`
- 168→6：`experiments/results/cross_variable_lag_adapter/l1_selection_168h_6h_smoke/`
- 联合复算：`experiments/results/cross_variable_lag_adapter/l1_selection_summary_smoke/`

## 5. GPU batch512 资源 smoke 阻断

已新增独立 `run_cross_variable_lag_l1_gpu_smoke.py`，固定168→6、F11、真实512样本完整 batch、真实最大结构和冻结F10基础；成功时记录设备、实际batch、峰值分配/保留显存、输入形状、梯度以及基础/PatchTST前后完整状态，并强制输出目录以`_gpu_smoke`结尾，不计算或解释性能。

本轮实际执行该命令时，环境中的 PyTorch 为 ROCm 7.14，但 `torch.cuda.is_available()` 为false、设备数为0，且容器没有 `/dev/kfd` 或 `/dev/dri`。脚本按设计报错中止，没有用CPU替代、没有伪造设备/显存记录，也没有创建 GPU smoke 目录。联合复算以`BLOCKED_NO_EXPOSED_GPU_DEVICE`记录该项，因此 `integrity_verification.json` 的总体`passed=false`；其余 CPU/数据/checkpoint 检查为true。

这构成当前唯一无法在本环境完成的资源验收阻断。获得暴露真实ROCm/CUDA设备的运行环境后，必须从现有替代168→6 F10 checkpoint执行完整512样本 F11 前向/反向，再不带`--allow-missing-gpu-smoke`重跑联合复算。不得在此之前启动正式实验。

## 6. 最终边界

相关 `py_compile` 通过，ST/L0/L1相关单元测试36/36通过。三个正式目录`l1_selection_24h_1h`、`l1_selection_168h_6h`、`l1_selection_summary`仍不存在；未commit、未push，原始CSV未修改。

P1冻结缺陷已修复，旧 CPU smoke 已明确作废并由从头运行的预提交工程 smoke 替换。但因真实GPU batch512资源 smoke 未完成，当前审计结论是 **工程修复通过、资源验收阻断、尚不允许正式启动**。

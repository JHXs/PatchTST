# L1 冻结实验协议

权威细节仍以理论阶段 03/04/05 为准；本文件把它们映射为实现验收项，不新增可调超参数。

## 数据与切分

- 北京 1013；原始 8760 个时间戳精确切为 6132/876/1752 行。
- 特征列只加载前 7008 行。test 特征、目标值、窗口、预测、缓存与指标计数必须恒为 0。
- train 目标时间严格落在 `[0,6132)`，validation 目标时间严格落在 `[6132,7008)`；test 只登记时间戳保留区间。三者目标时间集合两两无交。
- 站点候选池固定 1001–1036，含 1036、排除 1013。相关性阈值固定 Pearson `>0.85`；训练原始片完整性与相关性是唯一筛站输入。禁止 bfill。
- 中心辅助变量、顺序、pressure/零值/6 h 因果 ffill 与 10% 门直接复用 L0 实现。标准化器、场景阈值与筛站仅拟合训练片。

## 模型与训练

- 每任务/种子先训练 F00 degraded，再由其最佳 checkpoint 初始化 F10=Top-5+station bias，并冻结 PatchTST 主干。F10每轮调用整体train后必须立即强制整个PatchTST子模块eval；逐epoch、最佳checkpoint和最终加载均须逐元素保持F00初始化时的全部参数与buffers。
- F01 冻结 F00；F11/M、B1、B2、B-flat、B3 冻结并共享同一 F10 checkpoint。适配器训练期间完整基础模型保持 `eval`、`no_grad` 且状态哈希不变。
- M：`P=7,K=6,d=16`，原值+差分，全局 softmax 后污染物/气象组求和，`beta_max=0.2,beta_init=0.02`，输出末层零初始化。
- B1 用 PM2.5 自历史替换 42 个 token；B2 将每变量 lag0 复制到六个 lag 槽；B-flat 读取同一真实 84 维输入并固定 `w=15/47`；B3 固定均匀 gate。
- 所有适配器变体重置到同一种子后初始化，使用同一数据次序、MSE、AdamW(`lr=1e-3`,`weight_decay=1e-4`)、batch、ReduceLROnPlateau、早停与最佳验证 MSE 规则。
- 正式 24→1 为 40 epochs/patience8/batch256；168→6 为 30/patience6/batch512。smoke 只允许减少 epoch/样本，不改变结构、lag、门槛或优化器。
- 每次run manifest必须记录Git commit、dirty状态和关键L1源码SHA256；正式运行只接受clean tree。正式启动前另须完成168→6、F11、真实512样本完整batch的GPU前向/反向资源验收，只报告设备/显存事实。

## 指标与诊断

- 保存 validation 逐样本逐 horizon 的 scaled/物理量预测；由独立复算器重建 RMSE、MAE、SMAPE、bias、逐 horizon 和高污染指标。
- 2×2 固定计算 `G_A|S=0`、`G_A|S=1`、`G_S|A=0`、`G_comb`、`I` 和可定义时的 `R_attn`。
- 高污染由训练 PM2.5 的 q90 固定；快速上升/回落只作辅助诊断，分别由训练期逐小时变化 q90/q10 固定，普通状态为其余值。不得据验证结果改阈值。
- M-shuffle 对每个训练种子独立把验证起点分为完整 24 起点块，用 3101–3105 在块内共享置换完整七变量历史；原始与置换增益在同一删尾子集重算，禁止跨训练种子拼接。
- smoke 的 gate 状态固定为 `NOT_EVALUABLE_SMOKE`；不进入任何正式汇总。

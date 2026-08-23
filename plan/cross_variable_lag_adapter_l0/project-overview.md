# Cross-variable lag adapter L0 项目概览

- 分支：`experiment/cross-variable-lag-adapter-ablation`
- 阶段：S3 Experiments，仅执行 L0 选择区间筛查。
- 数据：北京 1013 原始逐小时时间戳；特征与目标只加载官方 train+validation 前 80% 行。
- 主任务：168→6；支持任务：24→1。
- 固定输入：`P=7`，顺序为 NO2、CO、O3、SO2、temperature、pressure_hPa_fixed、humidity_clean。
- 固定滞后：`K={0,1,2,3,6,12}`，lag 0 表示预测起点 `t` 的 `t-1` 观测。
- 禁止：MC-CVLRA 神经模块、正式 test 样本、原始 CSV 修改、后验改变量/lag/alpha/fold/阈值/清洗。
- 唯一阶段结论：168→6 硬门全部满足为 PASS，否则 STOP；24→1 不参与阶段门。

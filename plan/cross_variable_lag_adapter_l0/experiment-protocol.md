# L0 冻结实验协议

权威细节来自 `docs/第二创新点理论探索/03_最终推荐方案.md` 与 `05_数据语义审计与代理签字.md`；本文件只把本分支将执行的 L0 机械化。

- 先仅从全 CSV 读取 `time`，按原始时间戳行数建立 70%/10%/20% 边界；随后只加载前 80% 行的允许列。test 特征、目标、窗口和预测均不得构造。
- 北京中心站固定 1013；只加载 PM2.5 与冻结七变量，排除 PM10、wind_speed、weather、wind_direction。
- pressure 先执行 `<900 × 1.33322`，换算后域外 `[900,1100]` 标缺失；辅助污染物 `<=0`、humidity `0` 或 `>100`、temperature 域外 `[-50,60]` 标缺失。
- train 与 validation 分片独立执行最多 6 h 的因果 ffill；不 bfill。PM2.5 不修补。任一完整窗口有残余缺失即按三模型共同样本口径删除。
- 预测起点最早为 history；全部目标必须早于 test 起点。名义起点近等量切成 Q1–Q8，每个相邻边界从左块删除 `H-1` 个起点。
- 外折固定为 Q1–Q3→Q4 至 Q1–Q7→Q8。inner 按已 embargo 的 outer train 排序，前 `floor(0.85N)` 为名义 fit，随后再从 fit 尾部删除 `H-1` 个起点。
- outer、inner 和相邻 Q 边界均显式构造目标时间集合并断言交集为空。
- 特征：PM2.5 六 lag 的原值/一阶变化为共同控制；recent 加七变量 lag 0；lag 加七变量全部六 lag。三个 ridge 直接预测 PM2.5 目标。
- alpha 网格固定为 `{1e-4,1e-3,1e-2,1e-1,1,10,100}`。每个特征族/任务以 inner `RMSE_all` 选一个跨 horizon alpha；绝对差 `<=1e-12` 视为数值平局并取更大 alpha。每个 horizon 有独立系数。
- scaler 仅拟合 inner-fit 或 outer-train；ridge 仅拟合对应缩放后的训练片。所有拟合对象写入范围、行数、配置 hash 和状态 hash。
- 主门只看 168→6：`R_lag` 相对五折平均 `R_self` RMSE 改善至少 0.5%，相对 `R_recent` 至少 0.2%，且两项均至少 4/5 折同向。24→1 仅支持性报告。
- 所有输出必须标注“选择区间 L0，不是 test 或神经性能”。

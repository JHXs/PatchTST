# 第二创新点：频域分支图形数据清单

> 当前没有可用于论文的频域结果图。状态为“计划”的图只能在真实数据文件生成后绘制。

| Figure candidate | Required data | Purpose | Status |
|---|---|---|---|
| 训练期 PM2.5 频谱 | P0 训练期中心站序列和频谱导出 | 展示 24 h 等周期结构；只作动机，不作有效性证据 | 计划 |
| 固定/学习频率响应 | `frequency_diagnostics/band_masks*.csv` | 检查覆盖、边界移动、重叠和塌缩 | 计划 |
| 三种子容量控制配对图 | P1 `paired_differences.csv` | 比较 ST、时域适配器和固定频带 | 计划 |
| 动态门控分布 | `frequency_diagnostics/gates*.csv` | 展示样本变化、污染分组和气象分组 | 计划 |
| 上下文置乱干预 | P3 干预指标 | 检查动态门控是否使用真实上下文 | 计划 |
| 六步误差变化 | `per_horizon_metrics.csv` | 定位频域收益出现在哪些预测步 | 计划 |
| 峰值预测案例 | 锁定样本规则后的 `predictions/` | 展示峰值幅度和时序偏差；不得挑选最有利个例 | 计划 |
| 频域残差幅度 | `frequency_diagnostics/residuals*.csv` | 检查残差大小及是否越权主导 ST | 计划 |
| 五种子内部确认 | P6 `paired_differences.csv` | 展示逐种子方向、效应量和区间 | 计划 |
| 外部泛化对比 | 外部数据的配对汇总 | 展示已预注册数据集上的方向与幅度 | 计划 |
| rFFT–VMD 效率前沿 | P5 效率记录 | 比较精度收益和分解/推理成本 | 计划 |

图形脚本必须读取受版本控制的 CSV/JSON，不得在绘图代码中手写结果。峰值案例选择规则应在查看测试误差前固定。

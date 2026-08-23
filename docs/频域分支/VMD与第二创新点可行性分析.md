# VMD 与第二创新点可行性分析

> 当前定位（2026-08-23）：VMD 只作为传统时频分解对照，不作第二创新点的主方法。主方法见 [可学习的因果多尺度频域双分支方案](可学习的因果多尺度频域双分支方案.md)。

## 1. 结论

VMD 在技术上可以接入当前 ST-PatchTST，但“VMD＋PatchTST”本身不足以成为有说服力的第二创新点。原因是：

- VMD 需预先设定模态数 \(K\) 和带宽惩罚参数 \(\alpha\)；
- 逐窗口迭代求解的计算开销高于 rFFT 和可学习滤波器；
- 不同窗口可能出现模态混叠和编号交换；
- 对完整时间序列预先分解会将验证或测试期信息带入训练特征；
- 空气质量预测中已存在 VMD 与 GAT、BiLSTM、TCN 和注意力组合的研究。

因此，VMD 的最佳角色是：

> 在相同的锁定 ST 主干和预测端残差头下，与固定 rFFT 频带和受约束可学习频带比较精度、稳定性和效率。

## 2. 基本原理和参数边界

VMD 将信号表示为多个有限带宽模态之和：

\[
f(t)=\sum_{k=1}^{K}u_k(t).
\]

标准 VMD 的 \(K\) 仍需预先指定，并非完全自适应决定模态数。原始方法见 Dragomiretskiy 和 Zosso 的 [Variational Mode Decomposition](https://doi.org/10.1109/TSP.2013.2288675)。

当前项目不复制其他论文的 `K=9` 或 `alpha=2000`。如 VMD 进入对照，只允许在训练/验证阶段从预先限定的小集合中选择：

| 参数 | 预定义范围 |
|---|---|
| 历史长度 | 主设置 168 h |
| 模态数 \(K\) | 3、4、5 |
| 分解变量 | 仅中心站 PM2.5 |
| 排序 | 按每个窗口的中心频率升序 |
| 参数选择 | 仅使用训练和验证数据 |

24 h 窗口不扫描较大 \(K\)，也不作 VMD 主结论，因为样本点太少，容易过度分解和出现边界效应。

## 3. 严格滚动因果处理

对每个预测起点 \(t\)，VMD 只能接收当前历史窗口：

\[
u_1,\ldots,u_K=\operatorname{VMD}(X_c[t-L:t]).
\]

禁止：

```text
完整序列 → VMD → 切分训练/验证/测试
```

允许：

```text
先确定时间边界
  → 构造当前历史窗口
  → 仅分解该窗口
  → 预测未来原始 PM2.5 目标
```

VMD 输出可以按窗口和预定义参数缓存，但缓存 key 必须包含数据协议版本、窗口起点、历史长度、\(K\)、\(\alpha\) 和实现版本，防止复用旧协议的预计算模态。

## 4. 与当前 ST 的接入方式

不使用“模态重构后增强 PatchTST 输入”的旧思路。VMD 与可学习频带使用同一个预测端对照框架：

\[
h_k=E(u_k),
\qquad
R_{vmd}=\beta\tanh\!\left(H_{vmd}([w_kh_k]_{k=1}^{K})\right),
\]

\[
\widehat{Y}=\widehat{Y}_{\mathrm{ST}}+R_{vmd}.
\]

要求：

- 加载与主频域实验相同的锁定 ST 检查点；
- 冻结 PatchTST 和空间残差分支；
- 使用与 rFFT 变体尽可能一致的模态编码器、输出头、损失、优化器、早停和种子；
- 频域输出层零初始化，显式禁用后严格恢复锁定 ST 预测；
- 报告分解耗时、训练耗时、推理耗时和缓存存储量。

## 5. 进入顺序和停止条件

VMD 不参与第一轮频域可行性筛选。只有当下列条件均满足时，才实现 `st_rolling_vmd_residual`：

1. 无泄漏数据过渡基线通过；
2. 固定 rFFT 频带优于锁定 ST 和等容量时域适配器；
3. 受约束可学习频带已确定为主候选或确定无效；
4. VMD 的计算预算和缓存规则已在验证集上锁定。

如果滚动 VMD 没有超过固定 rFFT 频带，或其推理耗时增幅与精度收益不匹配，则作为负对照保留，不再扫描更多模态数和惩罚参数。

## 6. 相关工作与创新边界

需要在论文中明确区分下列已有工作：

- Dragomiretskiy and Zosso, [Variational Mode Decomposition](https://doi.org/10.1109/TSP.2013.2288675), IEEE TSP, 2014；
- Zhou et al., [FEDformer](https://proceedings.mlr.press/v162/zhou22g.html), ICML, 2022；
- Yi et al., [Frequency-domain MLPs are More Effective Learners in Time Series Forecasting](https://openreview.net/forum?id=iif9mGCTfy), NeurIPS, 2023；
- Yi et al., [FilterNet: Harnessing Frequency Filters for Time Series Forecasting](https://openreview.net/forum?id=ugL2D9idAD), NeurIPS, 2024；
- Ma et al., [MMFNet: Multi-Scale Frequency Masking Neural Network for Multivariate Time Series Forecasting](https://arxiv.org/abs/2410.02070), 2024 预印本；
- Wang et al., [Air quality forecasting using a spatiotemporal hybrid deep learning model based on VMD–GAT–BiLSTM](https://doi.org/10.1038/s41598-024-68874-x), Scientific Reports, 2024；
- Cheng and Zhang, [Forecasting of PM2.5 concentration based on variational mode decomposition and deep learning](https://doi.org/10.1038/s41598-026-56223-z), Scientific Reports, 2026。

因此，本项目不将“使用 FFT”“使用 VMD”或“学习频率掩码”单独写作创新。可能形成的贡献是：锁定已验证空间模型后的严格退化预测端频率残差、频率特异性容量控制、气象条件对照、峰值感知评估和无泄漏时间协议的组合证据。

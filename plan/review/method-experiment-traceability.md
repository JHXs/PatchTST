# 方法—实验追溯矩阵

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 成对动态邻站选择 | pairwise summary gate | degraded / center-only / pairwise | 报告表1 | 直接成对绝对值门控在当前任务未改善误差 | 已验证 |
| 无有效邻站时拒绝融合 | null neighbor | null weight 诊断 | 原始指标表 | 模型能分配一定概率给空邻站 | 已验证 |
| 空间信息真实性 | neighbor perturbation | 置零、样本打乱、中心复制 | 报告4.2和第5节 | 模型利用了少量真实邻站差异 | 已验证，效应较小 |
| 空间梯度修正 | neighbor-center delta value | 原始邻站值与差值残差 | 报告表1、表2 | 差值残差比绝对值残差更稳，平均RMSE改善 | 已验证 |
| 稳定性 | repeated seeds | 3 seeds | 均值标准差表 | 当前设置仍存在随机种子敏感性 | 已验证 |
| 避免输入污染 | forecast-stage bounded residual | 预测端密集/稀疏/站点偏置与退化基线 | 第二轮报告表2、表3 | 预测端残差在当前任务获得更明显的平均 RMSE 改善 | 已验证 |
| 严格空间归因 | true disable + global shuffle | 正常、禁用、置零、全局打乱、中心复制 | 第二轮报告表4 | 24→1改善依赖真实邻站及其时间对应关系 | 已验证；168→6仍有种子差异 |
| 站点长期先验 | learnable station gate bias | 无偏置与站点偏置 | 第二轮验证筛选和正式实验 | 站点偏置对24→1略有帮助，对168→6与密集版本相当 | 已验证 |
| 传播时滞 | training-only lag diagnostics | 0/1/2/3/6/12h相关与扩展窗口线性增量 | 第二轮报告表5 | 当前数据不支持把正滞后硬编码为主方案 | 已诊断 |
| 初始化稳定性 | degraded initialization + frozen backbone + Top-k selection | 五组互斥五种子确认 | 第二轮报告稳定性表 | 最终Top-5模型在两任务五个新种子上均降低RMSE并通过预注册实际收益门 | 已验证，严格门通过；范围限北京1013站PM2.5 |

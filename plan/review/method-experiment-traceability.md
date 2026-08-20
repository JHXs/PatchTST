# 方法—实验追溯矩阵

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 成对动态邻站选择 | pairwise summary gate | degraded / center-only / pairwise | 报告表1 | 直接成对绝对值门控在当前任务未改善误差 | 已验证 |
| 无有效邻站时拒绝融合 | null neighbor | null weight 诊断 | 原始指标表 | 模型能分配一定概率给空邻站 | 已验证 |
| 空间信息真实性 | neighbor perturbation | 置零、样本打乱、中心复制 | 报告4.2和第5节 | 模型利用了少量真实邻站差异 | 已验证，效应较小 |
| 空间梯度修正 | neighbor-center delta value | 原始邻站值与差值残差 | 报告表1、表2 | 差值残差比绝对值残差更稳，平均RMSE改善 | 已验证 |
| 稳定性 | repeated seeds | 3 seeds | 均值标准差表 | 当前设置仍存在随机种子敏感性 | 已验证 |

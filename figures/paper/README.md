# 论文图：Transformer 基线对比（方向 21）

数据来自 `tables/paper/PT*.csv`，由 `make_paper_baseline_artifacts.py` 生成（可重复）。

| 图 | 内容 | 放置建议 |
|---|---|---|
| `PF1_capacity_curve.{png,svg}` | 池化 RMSE 随可训练参数变化；黑星=锁定结构（冻结主干、18 站），红=Informer（4 个容量点，匹配点 5,125 已标注），蓝/绿=单站点 GRU/LSTM 曲线，紫= TST 逐 (L,H) 散点（非容量对齐） | 正文（基线对比主图） |
| `PF2_per_lead.{png,svg}` | 我们相对 Informer 各臂的逐 lead 优势（H=6/12/24；正值=我们更好） | 正文或附录（说明优势集中在短预测步） |

注意：X 轴跨约 200–50,000 可训练参数（对数刻度）；图注必须声明基线与我们的信息集差异。

# 论文口径表（Transformer 基线，方向 21）

由 `make_paper_baseline_artifacts.py` 从 `source/` 下的源 CSV（复制自
`experiment/baseline-comparison-v2-ablation:tables/baseline_v2/`）后处理生成，**不含任何新训练**。

```bash
python make_paper_baseline_artifacts.py --source tables/paper/source \
  --out-tables tables/paper --out-figures figures/paper
```

| 表 | 用途 | 建议放置 |
|---|---|---|
| `PT1_informer_main` | Informer 各容量 vs 我们（含匹配点与双口径） | 正文主表 |
| `PT2_capacity_points` | 容量曲线数据（含 GRU/LSTM 复用曲线、TST 逐 (L,H)） | 正文配图数据 |
| `PT3_dual_criterion` | 验证集口径 / 测试集口径 | 附录（披露义务） |
| `PT4_per_lead` | 逐 lead 对比 | 正文配图数据 / 附录 |
| `PT5_rank_correlation` | 验证损失–测试 RMSE 秩相关 | 附录（方法学） |
| `PT6_compliance_self_check` | 十项合规自检 | 附录（复现声明） |

关键口径：基线=单站点；我们=18 站；配对只用共同子集；负值=我们更差（`*_our_advantage*` 列为正=我们更好）。

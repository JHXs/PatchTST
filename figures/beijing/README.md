# 北京 1013 第一创新论文图表

数据来源：稳定 `main` 上已合并的第一创新结果，`experiments/results/st_patchtst_ablation/`。
最终确认尝试为 `frozen_topk5_2047_2051`（变体 `st_sparse_station_bias_delta_forecast`，k=5），
严格放行记录见 `experiments/results/st_patchtst_ablation/stability_release_status.json`。

本目录不含任何新训练：全部表格与图由已提交的 CSV 与训练日志后处理得到。

## 重新生成

```bash
python make_beijing_paper_artifacts.py \
  --results experiments/results/st_patchtst_ablation \
  --out-tables tables/beijing --out-figures figures/beijing
```

## 表格

| 表 | 内容 | 建议放置 |
|---|---|---|
| `B1_final_gates` | 最终确认：两任务的 base/候选 RMSE、均值±std、Bootstrap CI95、改善种子数与精确符号 p、门槛与严格门 | 主结果表 |
| `B2_final_per_seed` | 最终确认逐种子配对（种子 2047–2051） | 附录 |
| `B3_all_attempts` | **全部五轮尝试**的方向门、实际收益门与严格门状态 | 主结果或附录（诚信要求） |
| `B4_neighbor_interventions` | 最终轮的邻站干预消融：正常 / 显式禁用 / 置零 / 全局打乱 / 中心复制 | 机制有效性证据 |
| `B5_gate_diagnostics` | alpha、空间残差比例、门控熵、空邻站权重、门控方差、权重范数 | 附录 |
| `B6_topk_validation`、`B6b_topk_scores` | 2024–2026 验证段上的 Top-k 选择 | 方法节 |
| `B7_lag_diagnostics` | 邻站传播滞后诊断（0 h 最优，其余为负结果） | 附录 |

## 图

| 图 | 内容 | 建议放置 |
|---|---|---|
| `BF1_final_per_seed` | 最终确认逐种子配对降幅与预注册阈值 | 主结果图 |
| `BF2_attempts` | 五轮尝试的均值±std，按**严格门**着色并标注改善种子数 | 主结果图（必须保留失败轮次） |
| `BF3_interventions` | 邻站干预对 RMSE 的影响（含退化基线参考线） | 机制有效性图 |
| `BF4_topk_selection` | 验证段 Top-k 选择曲线与选中 k | 方法图 |
| `BF5_lag_diagnostics` | 滞后 RMSE 增益曲线（标注改善站点数） | 附录（负结果） |
| `BF6_training_curves` | 最终轮训练/验证损失曲线（5 种子） | 附录 |

每张图输出 `.pdf`（矢量）与 `.png`（300 dpi）。

## 建议图注

- BF1: *Per-seed paired RMSE reduction of the frozen structure over the degraded PatchTST baseline (seeds 2047–2051); dashed line: pre-registered practical threshold.* /
  固定结构相对退化 PatchTST 的逐种子配对 RMSE 降幅（种子 2047–2051）；虚线为预注册实际收益门。
- BF2: *All five confirmation attempts. Bars are coloured by the pre-registered strict gate (direction consistency + practical effect); labels give improved seeds / runs. Earlier failures are shown deliberately.* /
  五轮确认尝试；柱色表示预注册严格门（方向一致性＋实际收益）是否通过，标注为改善种子数；失败轮次有意保留。
- BF3: *Neighbour reliance: RMSE of the frozen spatial branch with neighbours disabled, zeroed, shuffled or replaced by the centre series (mean ± sd over 5 seeds); dashed line: degraded baseline.* /
  邻站依赖：禁用、置零、打乱或替换为中心序列后的 RMSE（5 种子均值±标准差）；虚线为退化基线。
- B4 表补充说明：最终轮 24→1 下，打乱邻站使 RMSE 上升 9.41%、置零 7.13%、显式禁用 3.24%、中心复制 3.13%，退化基线不受任何干预影响（恒为 0 变化），说明空间收益确实来自邻站信息而非参数化副作用。

## 使用注意

- `B3`/`BF2` 必须与最终确认同时展示：本项目要求不得只报告有利轮次。前四轮失败的原因为方向门（168→6 有 4/5 种子改善）或实际收益门（0.29%、0.497% 低于 0.5%）。
- 最终确认的五轮尝试使用了不同种子块（2027–2031、2032–2036、2037–2041、2042–2046、2047–2051），其中 2027–2046 已在探索中使用，论文中不得把全部五轮合并当作独立证据。
- 结论范围仍限定为北京 1013、PM2.5、24→1 与 168→6；跨城市泛化的材料见 `figures/round14/`。
- 仓库内没有北京预测文件（npz 未入库），因此本目录不提供示例时序图；如需，须单独授权重跑。

## 结构消融（2026-09-14 补充）

| 表/图 | 内容 | 说明 |
|---|---|---|
| `tables/beijing/B8_architecture_ablation` | 架构消融：centre-only / pairwise / pairwise+delta（输入端）/ 预测端各变体 / 生产结构 | **全部为 test 划分**；含 `split` 列 |
| `tables/beijing/B9_dev_selection_validation` | 开发选择轮（rounds 3–6）与 k 扫描（k=3/5/7/9/12） | **valid 划分**，仅用于结构选择，不与 B8 混合 |
| `figures/beijing/BF7_structure_ablation` | 上排：各变体相对同目录退化基线的配对降幅（标注改善种子数）；下排：邻站禁用/打乱的 RMSE 增幅 | 灰色＝端到端 dev（test），橙色＝冻结生产（最终确认） |

**使用注意（重要）**

- 评估划分必须区分：rounds 1–2 与确认轮使用 **test**；rounds 3–6 的选择运行使用 **valid**。两者绝对 RMSE 不可比（同协议下 valid 基线约 23.87 而 test 基线约 21.29），因此 B8 只收录 test 行，B9 单独列出 valid 行。
- B8 中的降幅是"同目录、同种子、同协议"的配对结果，跨行比较只在同一 block 内成立；`frozen_production` 行来自最终确认（种子 2047–2051，5 种子），其余为 dev（种子 2024–2026，3 种子）。
- 输入端门控（centre-only、pairwise）在 24→1 上为负（−0.33%、−0.34%，改善 1/3 与 0/3），这是真实结论，论文中不得省略。
- 下排的邻站依赖只对含空间分支的变体有定义；退化基线对任何邻站干预恒定不变（0%）。

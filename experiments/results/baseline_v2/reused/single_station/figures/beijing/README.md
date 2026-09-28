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

## 结构消融（2026-09-15 重做：匹配的两种协议 × 同一条组件链）

| 表/图 | 内容 |
|---|---|
| `tables/beijing/B8_architecture_ablation` | 匹配组件消融：9 个臂 × 2 种训练协议 × 2 个任务，全部 test 划分、种子 2024–2026 |
| `tables/beijing/B9_dev_selection_validation` | 早期开发选择轮与 k 扫描（**valid 划分**），仅用于结构选择 |
| `figures/beijing/BF7_structure_ablation` | 上排：各变体配对降幅（标注改善种子数）；下排：打乱邻站后的 RMSE 增幅 |

**为什么重做**：此前的 B8 把不同轮次、两种协议、两种评估划分（valid/test）的结果拼在一起，无法做组件归因。现在每个 (协议, 任务) 是一次运行、含完全相同的九个臂，退化臂在两种协议下训练相同，因此两种协议共享同一基线，所有降幅都是同 run 同种子配对。

**核心结论（可直接写进论文）**

- `centre-only` 与 `pairwise`（输入端门控）在两种协议、两个任务上基本为 0 或负：24→1 端到端 −0.58%（1/3）、冻结 +0.02%（2/3）；168→6 端到端 **−2.31%（0/3）**、冻结 −0.26%（0/3）。这解释并支持"把融合放在预测端"的设计选择。
- 预测端变体在 24→1 上稳定为正（+2.4%~+3.3%），冻结协议下 `sparse + station-bias` 最好（+3.28%，3/3）。
- **组件排序依赖训练协议**：168→6 端到端协议下 `sparse + delta（预测端）` 最好（+1.56%，3/3），而冻结协议下 `pairwise + delta（输入端）` 最好（+2.25%，3/3），预测端变体仅 +0.42%~+0.67%。两种协议不可合并报告。
- 邻站依赖（下排）：冻结协议下打乱邻站使 24→1 的 `pairwise + delta（输入端）` RMSE 上升 22.1%，其他预测端变体上升 7%~11%，端到端协议下同一变体仅 0.01%——说明冻结主干后模型对邻站输入的依赖显著增强。

**使用注意**

- B8 全部为 test 划分；B9 为 valid 划分，二者不可混比（同协议下 valid 基线 23.87 vs test 基线 21.52）。
- 本消融为探索性（3 种子、选择阶段），**不是**确认证据；确认证据仍是 5 种子预注册门（`stability_confirmation_topk5_*`）。
- 原始运行（含 checkpoints/predictions）归档：`frozen-protocol-component-ablation-2024-2026.tar.zst`，SHA-256 `14a1b80f…`。

## 示例时序与可复现性核对（2026-09-14）

为补齐论文的示例时序图，按确认轮**完全相同的配置**重跑了两个任务（种子 2047–2051）：

```bash
python run_st_patchtst_ablation.py --history 24 --horizon 1 \
  --seeds 2047,2048,2049,2050,2051 \
  --variants degraded_patchtst,st_sparse_station_bias_delta_forecast \
  --epochs 40 --patience 8 --batch-size 256 --sparse-neighbor-top-k 5 \
  --evaluation-split test --initialize-from-degraded --freeze-backbone \
  --device cuda:0 --output-dir <rerun-24h-dir>

python run_st_patchtst_ablation.py --history 168 --horizon 6 \
  --seeds 2047,2048,2049,2050,2051 \
  --variants degraded_patchtst,st_sparse_station_bias_delta_forecast \
  --epochs 30 --patience 6 --batch-size 512 --sparse-neighbor-top-k 5 \
  --evaluation-split test --initialize-from-degraded --freeze-backbone \
  --device cuda:0 --output-dir <rerun-168h-dir>
```

| 表/图 | 内容 |
|---|---|
| `tables/beijing/B10_reproducibility_rerun_vs_recorded` | 重跑 vs 记录的 RMSE/MAE/best_epoch（20 行） |
| `figures/beijing/BF8_example_series_1013_24x1` | 北京 1013 站 24→1 示例（种子 2047，test 划分） |
| `figures/beijing/BF8_example_series_1013_168x6_lead1` | 同站 168→6 的 lead-1 序列示例 |

**可复现性结论**：两个任务、5 个种子、2 个实验臂共 20 次运行全部与记录值**逐位一致**（RMSE/MAE 绝对差 0.0000，best_epoch 全部相同）。

**示例窗口规则（避免挑选）**：取 test 划分中观测序列标准差最大的 120 小时窗口（即波动最强的污染过程），规则固定、可由脚本复算。

**定位声明**：示例重跑仅用于绘图与可复现性核对，**不是**新的确认证据；确认证据仍是 `stability_confirmation_topk5_*`（种子 2047–2051）五轮原始运行。重跑产物（predictions/raw_metrics/config，不含 checkpoints）归档：

```text
/home/hansel/.herdr/artifacts/PatchTST/beijing-example-rerun-2047-2051.tar.zst
SHA-256: 8ad7fe99d50f79b44acdc74eff6f94812b75e42a01d7d8d2b56d47906c5645ec
size: 924639 bytes
```

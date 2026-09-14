# Round 14 论文图表（跨城市泛化确认）

数据来源：`experiment/cross-city-generalization-ablation@5211d7b` 的正式确认运行（2026-09-14）。

- 执行 commit：`71bb1d952a995900f7655894acec72211badc958`
- 摘要证据：`experiments/results/round14/cross_city_generalization_summary/`
- 全量产物归档（含 160 个预测 npz、checkpoints、逐站 metadata）：`/home/hansel/.herdr/artifacts/PatchTST/round14-cross-city-generalization-71bb1d9.tar.zst`，SHA-256 `41be51f36092b58322455fb9848ca8140a676cb51c170bba6a18c893c1aa3e5e`

## 重新生成

```bash
python make_round14_paper_artifacts.py \
  --summary-dir experiments/results/round14/cross_city_generalization_summary \
  --archive /home/hansel/.herdr/artifacts/PatchTST/round14-cross-city-generalization-71bb1d9.tar.zst \
  --out-tables tables/round14 --out-figures figures/round14
```

脚本只做后处理与绘图，不训练、不重新选择任何结构；`F8`/`T8`/`T9` 需要 `--archive`（从归档读取预测与 metadata），其余仅依赖已提交的 summary CSV。

## 表格

| 表 | 内容 | 建议放置 |
|---|---|---|
| `T1_main_gates` | 预注册 G1–G4 主结果与判定（两任务） | 主结果表 |
| `T2_per_station` | 逐站 base/ST 的 RMSE、MAE、SMAPE 与降幅 | 主结果表或附录 |
| `T3_per_lead` | 逐预测步 RMSE/MAE/SMAPE（168→6 六步，24→1 一步） | 主结果或附录 |
| `T4_per_block` | 五个确认块的稳健性 | 附录 |
| `T5_peak_subset` | 各站训练段第 90 百分位以上高浓度子集 | 附录 |
| `T6_headline_metrics` | 两臂池化 RMSE/MAE/SMAPE 与元素数 | 主结果表（摘要行） |
| `T7_smape_quantile_decomposition` | 按训练段真实值分位的 SMAPE 分解 | 附录 |
| `T8_top5_neighbours` | 各中心站在 C-fit 段选出的 Top-5 邻站及相关性 | 方法/数据节 |
| `T9_data_coverage` | 共同时间轴与 C-fit/C-val/C-confirm 行数、候选数 | 数据节 |

每张表同时输出 `.csv`、`.md` 与（主表）`.tex`。

## 图

| 图 | 内容 | 建议放置 |
|---|---|---|
| `F1_per_station_reduction` | 逐站配对 RMSE 降幅（条：5 种子均值；线：min–max），两任务同 y 轴 | 主结果图 |
| `F2_per_lead` | 168→6 逐预测步 RMSE（base vs ST）与降幅 | 主结果图 |
| `F3_per_block` | 五个确认块的降幅稳健性 | 附录 |
| `F4_pair_distribution` | 每个任务全部 40 个 (站, 种子) 配对的降幅分布 | 主结果或附录 |
| `F5_rmse_scatter` | 逐站 base vs ST RMSE 散点（对角线上方为改善） | 附录 |
| `F6_smape_quantiles` | 按真实值分位的 SMAPE（base vs ST） | 附录 |
| `F7_peak_subset` | 高浓度子集 RMSE（逐站） | 附录 |
| `F8_example_series_9033_24x1` | 示例时序：观测、两臂预测与空间修正量（首 120 h） | 方法/案例图 |

每张图输出 `.pdf`（矢量，投稿用）与 `.png`（300 dpi，预览用）。

## 建议图注（英文 / 中文）

- F1: *Per-station paired RMSE reduction of the frozen spatial structure over the same-seed degraded PatchTST baseline on Guangzhou (8 unconsumed centre stations, 5 seeds). Bars: mean over seeds; vertical lines: min–max. Panels share the y-axis.* /
  固定空间结构相对同种子退化 PatchTST 的逐站配对 RMSE 降幅（广州 8 个未消费中心站、5 个种子）。柱为种子均值，竖线为 min–max，两任务共用 y 轴。
- F2: *Error by forecast lead for 168→6 (mean over 8 stations × 5 seeds).* /
  168→6 任务逐预测步误差（8 站 × 5 种子均值）。
- F4: *All 40 station–seed pairs per task; black bars mark the pre-registered pool effect.* /
  每任务全部 40 个站–种子配对的降幅分布；黑杠为预注册池效应。
- F8: *Example 24→1 forecast at Guangzhou station 9033 (seed 7001, first confirmation block). Lower panel shows the additive spatial correction.* /
  广州 9033 站 24→1 示例（种子 7001，第一个确认块）；下方面板为空间修正量。

## 使用注意

- `T1` 中的符号检验尾概率是**描述性**诊断：同一站点的 5 个种子与相邻时间块并不独立，不得据此声称独立显著性；放行判定由 G1–G4 的合取门给出。
- `T2` 显示部分站点（如 9020）SMAPE 上升而 RMSE 改善——这是真实结果，写论文时不得只报池化 SMAPE；`T6`/`T7` 给出池化与分位分解。
- 本目录图表只支持“第一创新的跨城市泛化确认”这一结论，不支持“第二创新机制”；也不得称为全新城市或完全盲测（广州 9022、9027 已有历史消费；b2/b3 保留池未被使用）。

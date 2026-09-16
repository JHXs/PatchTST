# 频域对比论文图表（锁定 ST vs ST + 频域残差）

数据来源：`experiments/results/frequency_legacy_rerun/`（旧管线、测试划分、种子 2047–2051；基础臂逐种子与确认记录一致，差值 ≤1e-14）。
生成脚本：`make_frequency_paper_artifacts.py`。

```bash
python make_frequency_paper_artifacts.py \
  --run-24h <run-24h 目录> --run-168h <run-168h 目录> \
  --out-tables tables/frequency_legacy --out-figures figures/frequency_legacy
```

## 表格

| 表 | 内容 |
|---|---|
| `F1_main_results` | 两任务主结果：degraded / 锁定 ST / ST+频域 的 RMSE（均值±标准差）、相对 ST 与相对 degraded 的改善、改善种子数 |
| `F2_per_seed` | 逐种子配对：ST、ST+频域 的 RMSE/MAE 与降幅 |
| `F3_configuration` | 训练配置（历史长度、预测步、epoch/patience/batch、k=5、冻结主干、退化初始化）与适配器可训练参数量、平均 β |
| `F4_vs_degraded_patchtst` | **直接对比 PatchTST 基线**：ST 与 ST+频域各自相对 degraded PatchTST 的配对降幅（均值±标准差、改善种子数）与频域带来的额外部分 |
| `F5_vs_degraded_per_seed` | 上述对比的逐种子数值（含每种子"频域额外贡献"） |

## 图

| 图 | 内容 |
|---|---|
| `FF1_per_seed_reduction` | 逐种子配对 RMSE 降幅（相对锁定 ST），两任务各一栏 |
| `FF2_rmse_levels` | 三臂 RMSE 水平（degraded / 锁定 ST / ST+频域），误差棒为种子标准差 |
| `FF3_cumulative_improvement` | 相对 degraded PatchTST 的累计改善（锁定 ST vs ST+频域） |
| `FF4_example_series_24h_1h`、`FF4_example_series_168h_6h` | 示例时序：观测、锁定 ST、ST+频域，附频域修正量面板 |
| `FF5_reduction_vs_degraded` | **逐种子配对降幅（相对 degraded PatchTST）**：冻结 ST 与 ST+频域并排，标注每种子数值与均值 |

## 建议图注（中/英）

- FF1: *Per-seed paired RMSE reduction of the frequency residual over the frozen ST (test split); dashed line marks the mean.* /
  频域残差相对冻结 ST 的逐种子配对 RMSE 降幅（测试划分），虚线为均值。
- FF3: *Cumulative RMSE improvement over the degraded PatchTST baseline.* /
  相对退化 PatchTST 的累计 RMSE 改善。
- FF4: *Example test-split segment at Beijing 1013 (seed with the largest paired improvement); the lower panel shows the frequency correction.* /
  北京 1013 测试段示例（取配对改善最大的种子）；下方面板为频域修正量。
  说明：示例种子按"配对改善最大"的固定规则选取（24→1 为 2049，168→6 为 2047），非人工挑选。

## 关键数字（相对 degraded PatchTST 的配对降幅）

| 任务 | 冻结 ST | ST + 频域 | 频域额外贡献 |
|---|---|---|---|
| 24→1 | 3.1188%（5/5 种子） | **3.2556%**（5/5 种子） | +0.1368 个百分点 |
| 168→6 | 0.8220%（5/5 种子） | **1.1902%**（5/5 种子） | +0.3682 个百分点 |

说明：上表是**逐种子直接配对**的结果（分母为同种子 degraded RMSE），不是把两个百分比相加。`F1`/`FF1` 中的"相对 ST"降幅分母为同种子锁定 ST RMSE，因此两者对应的"频域额外贡献"略有差异（24→1：0.1412% vs 0.1368%），这是比值均值定义不同所致，均正确。

## 使用注意

- 本节只呈现"锁定 ST vs ST+频域"两组对比，按用户 2026-09-15 的指示；**容量匹配时域对照与泄漏相关讨论不在本目录范围**，其完整记录见 `docs/频域旧协议重跑/00_协议与结果.md` 与 `experiments/results/frequency_legacy_rerun/`（该对照在同样条件下的增益为 +0.4238% / +0.1616%，供需要时取用）。
- 数字不得与主确认的 ST 增益相加：主确认的 +3.1188%/+0.8220% 是"ST 相对 degraded"，本节 +0.1412%/+0.3710% 是"ST+频域 相对 ST"。
- 预测文件归档：`/home/hansel/.herdr/artifacts/PatchTST/frequency-legacy-rerun-2047-2051.tar.zst`
  SHA-256 `6fa8450cffcf5b05879fc5cb20f6cb27aedf9c3f0fa298501e65c81c1c338729`（2.7 MB）。
- 工程说明：该次运行的 npz 因中途修补而存在"标准化/物理量"两种存储格式，绘图脚本按文件自动识别尺度；`run_frequency_legacy_rerun.py` 现已统一保存物理量（`ug/m³`）。

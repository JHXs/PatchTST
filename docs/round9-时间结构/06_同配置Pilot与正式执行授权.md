# 第 9 轮 CCSPRF 同配置 Pilot 与正式执行授权

日期：2026-09-10。

## Pilot 边界

在固定实现 `2c5b071` 上，使用非正式种子 `991105` 运行与正式 L0 相同的 `8736` 点、两任务、两条件、四臂、PatchTST/optimizer/epoch/patience 配置，仅用于工程完整性和成本估算。输出：`/tmp/ccsprf-l0-pilot-formal-shape-991105`。该 seed 不属于正式 `20260939..20260968`，pilot RMSE 不参与结构、DGP、tau、门或性能结论。

执行结果：16/16 key 完成，进程返回 0，墙钟 194 秒；逐 key 记录的训练时间合计 `189.6006s`，中位 `9.6938s`，最大 `29.5017s`。其中 24→1 八个 key 合计 `37.5912s`，168→6 八个 key 合计 `152.0094s`。

独立执行 `summarize_ccsprf_l0.py` 后状态为 `NOT_APPLICABLE_FIXTURE`，DGP、split、scale、唯一时间戳 LOO、target、calendar/global/EMB prior、预测重建、参数、manifest/progress 和指标复算全部通过。输出约 `3.0M`、63 个文件。

## 正式资源估算

正式矩阵是 pilot 的 30 倍：480 key。按 pilot 逐 key 实测线性估算训练时间约 `5688s`（94.8 分钟）；考虑数据/先验写入、进程和独立摘要开销，预留约 1.5–2 小时。磁盘按 pilot 线性上界约 90 MB。当前两张 ROCm 设备可见，runner 单进程固定使用 `cuda`，不在正式运行中改为多进程或调整 batch/epoch。

## 主代理授权

工程门与资源门 **PASS**。授权在提交本记录后、源码 clean 的固定 commit 上启动正式 30 种子 L0：

```text
/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python run_ccsprf_l0.py \
  --formal --device cuda \
  --output-dir experiments/results/round9/l0_formal
```

如中断，只允许使用同 commit 和同目录执行 `--formal --resume`；不得删除或覆盖正式输出。完成后必须先运行独立 summarizer，再判断 PASS/STOP。任何完整性失败属于 `ENGINEERING_FAILURE`，不得作为性能 STOP；任何性能门失败立即 STOP，不调参挽救，不进入真实站点数据。

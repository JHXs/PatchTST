# 匹配组件消融（两种训练协议 × 同一条组件链）

目的：回答"在最终采用的训练制度下，各组件各自贡献多少"，并避免此前把不同轮次、不同协议、
不同评估划分的结果拼在同一张消融表里。

## 设计

- 每个 (协议, 任务) 是**一次运行**，包含完全相同的九个臂：`degraded_patchtst` + 八个空间变体。
- 协议：`end-to-end`（不冻结主干、不从退化基线初始化）与 `frozen + degraded init`（最终生产配置）。
- 种子：2024、2025、2026（与早期开发轮一致）。
- 任务：24→1（40 epoch / patience 8 / batch 256）与 168→6（30 / 6 / 512）；k=5；`evaluation-split test`。
- 退化臂在两种协议下训练完全相同，因此两种协议共享同一基线与同一基线检查点；每条降幅都是
  同 run、同种子的配对结果。

## 运行命令

```bash
# 端到端协议
python run_st_patchtst_ablation.py --history 24 --horizon 1 --seeds 2024,2025,2026 \
  --variants degraded_patchtst,st_center_only,st_pairwise,st_pairwise_delta,st_pairwise_delta_forecast,st_sparse_delta_forecast,st_confident_delta_forecast,st_station_bias_delta_forecast,st_sparse_station_bias_delta_forecast \
  --epochs 40 --patience 8 --batch-size 256 --sparse-neighbor-top-k 5 \
  --evaluation-split test --device cuda:0 --output-dir <end-to-end-24h>

python run_st_patchtst_ablation.py --history 168 --horizon 6 --seeds 2024,2025,2026 \
  --variants <同上九个> --epochs 30 --patience 6 --batch-size 512 --sparse-neighbor-top-k 5 \
  --evaluation-split test --device cuda:0 --output-dir <end-to-end-168h>

# 冻结生产协议：在端到端命令上追加 --initialize-from-degraded --freeze-backbone
```

## 定位

探索性消融（3 种子、结构选择阶段），**不是**确认证据；确认证据仍是
`stability_confirmation_topk5_*`（种子 2047–2051，5 种子，预注册门）。

## 原始运行归档

```text
/home/hansel/.herdr/artifacts/PatchTST/frozen-protocol-component-ablation-2024-2026.tar.zst
SHA-256: 14a1b80f4fabac8879bab2bf5613b0ef549a6df98bd2a344fed7b3f9a2101d7c
size: 11260710 bytes
```

#!/usr/bin/env bash
# 阶段 A2：追加历史窗口 L=48 与 L=72（H=1,3,6,12,24，种子 2047–2051）。
# 协议与阶段 A 完全一致；训练预算按历史长度分档：L<=48 用 40/8/256，L>48 用 30/6/512。
set -u
PY=/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python
OUT=experiments/results/horizon_coverage
for L in 48 72; do
  for H in 1 3 6 12 24; do
    DIR="$OUT/${L}h_${H}h"
    if [ -f "$DIR/raw_metrics.csv" ]; then echo "已完成 $DIR"; continue; fi
    if [ "$L" -le 48 ]; then EPOCHS=40; PATIENCE=8; BATCH=256; else EPOCHS=30; PATIENCE=6; BATCH=512; fi
    echo "=== 运行 ${L}h→${H}h (epochs=$EPOCHS patience=$PATIENCE batch=$BATCH) ==="
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl-horizon "$PY" run_st_patchtst_ablation.py \
      --history "$L" --horizon "$H" --seeds 2047,2048,2049,2050,2051 \
      --variants degraded_patchtst,st_sparse_station_bias_delta_forecast \
      --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$BATCH" \
      --sparse-neighbor-top-k 5 --evaluation-split test \
      --initialize-from-degraded --freeze-backbone --device cuda:0 \
      --output-dir "$DIR" || { echo "失败: $DIR"; exit 1; }
  done
done
echo STAGE_A2_DONE

#!/usr/bin/env bash
# 阶段 A：北京 1013 的降级 PatchTST 与冻结 ST，在 8 个新任务配置上，种子 2047–2051。
# 旧管线、测试划分、k=5、冻结主干、退化初始化——与主确认完全一致。
set -u
PY=/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python
OUT=experiments/results/horizon_coverage
for L in 24 168; do
  for H in 1 3 6 12 24; do
    if [ "$L" = "24" ] && [ "$H" = "1" ]; then echo "跳过 24h_1h（已有确认结果）"; continue; fi
    if [ "$L" = "168" ] && [ "$H" = "6" ]; then echo "跳过 168h_6h（已有确认结果）"; continue; fi
    DIR="$OUT/${L}h_${H}h"
    if [ -f "$DIR/raw_metrics.csv" ]; then echo "已完成 $DIR"; continue; fi
    if [ "$L" = "24" ]; then EPOCHS=40; PATIENCE=8; BATCH=256; else EPOCHS=30; PATIENCE=6; BATCH=512; fi
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
echo STAGE_A_DONE

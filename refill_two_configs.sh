#!/usr/bin/env bash
# 补跑被阶段 A 跳过的两个配置（24h_1h、168h_6h），以便阶段 B 能覆盖全部 20 个配置。
# 协议与主确认完全一致；跑完后会与确认记录逐种子比对。
set -u
PY=/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python
OUT=experiments/results/horizon_coverage
run_one() {
  local L=$1 H=$2 E=$3 P=$4 B=$5
  local DIR="$OUT/${L}h_${H}h"
  if [ -f "$DIR/raw_metrics.csv" ]; then echo "已有 $DIR"; return 0; fi
  echo "=== 运行 ${L}h→${H}h ==="
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl-horizon "$PY" run_st_patchtst_ablation.py \
    --history "$L" --horizon "$H" --seeds 2047,2048,2049,2050,2051 \
    --variants degraded_patchtst,st_sparse_station_bias_delta_forecast \
    --epochs "$E" --patience "$P" --batch-size "$B" \
    --sparse-neighbor-top-k 5 --evaluation-split test \
    --initialize-from-degraded --freeze-backbone --device cuda:0 \
    --output-dir "$DIR" || return 1
}
run_one 24 1 40 8 256 && run_one 168 6 30 6 512 && echo REFILL_DONE

#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
unset DISPLAY WAYLAND_DISPLAY

PY=/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python
"$PY" run_endtoend_spatial.py \
    --configs 24x24 \
    --seeds 2048 \
    --backbones gru_h8 \
    --device cuda \
    --output-root experiments/results/endtoend_spatial_debug_nonfinite

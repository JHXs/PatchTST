#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
unset DISPLAY WAYLAND_DISPLAY

PY=/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python
MODE=${1:-all}
DEVICE=${DEVICE:-cuda}

case "$MODE" in
  test)
    "$PY" -m py_compile run_trainable_matched_baselines.py summarize_trainable_matched.py test_trainable_matched.py
    "$PY" -m unittest -v test_trainable_matched.py
    ;;
  smoke)
    "$PY" run_trainable_matched_baselines.py --smoke --device "$DEVICE"
    ;;
  full)
    "$PY" run_trainable_matched_baselines.py --device "$DEVICE"
    ;;
  summarize)
    "$PY" summarize_trainable_matched.py
    ;;
  all)
    "$0" test
    "$0" smoke
    "$0" full
    "$0" summarize
    ;;
  *)
    echo "usage: $0 {test|smoke|full|summarize|all}" >&2
    exit 2
    ;;
esac

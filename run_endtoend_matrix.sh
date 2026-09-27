#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
unset DISPLAY WAYLAND_DISPLAY

PY=/home/hansel/Documents/ITProject/Python/PatchTST/.venv/bin/python
resume_from=${RESUME_FROM:-}
resume_reached=false
if [[ -z "$resume_from" ]]; then
    resume_reached=true
fi

for history in 24 48 72 168; do
    for horizon in 1 3 6 12 24; do
        task="${history}x${horizon}"
        if [[ "$task" == "$resume_from" ]]; then
            resume_reached=true
        fi
        if [[ "$resume_reached" != true ]]; then
            continue
        fi
        seeds="2047 2048 2049"
        if [[ "$history" == "24" && "$horizon" == "1" ]] || \
           [[ "$history" == "168" && "$horizon" == "6" ]]; then
            seeds="2047 2048 2049 2050 2051"
        fi
        for seed in $seeds; do
            for backbone in gru_h8 gru_h16 gru_h32 gru_h64 lstm_h16 lstm_h32; do
                # ROCm recurrent kernels can poison later runs in the same
                # process after an asynchronous fault.  One identity per
                # process keeps failures attributable and remains resumable.
                "$PY" run_endtoend_spatial.py \
                    --configs "${history}x${horizon}" \
                    --seeds "$seed" \
                    --backbones "$backbone" \
                    --device cuda
            done
        done
    done
done

"$PY" summarize_endtoend_spatial.py

#!/usr/bin/env bash
set -euo pipefail

# Attempt 1: joint station-bias training, seeds 2027-2031.
uv run python run_st_patchtst_ablation.py --history 24 --horizon 1 --epochs 40 --patience 8 --seeds 2027,2028,2029,2030,2031 --variants degraded_patchtst,st_station_bias_delta_forecast --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_24h_1h
uv run python run_st_patchtst_ablation.py --history 168 --horizon 6 --batch-size 512 --epochs 30 --patience 6 --seeds 2027,2028,2029,2030,2031 --variants degraded_patchtst,st_station_bias_delta_forecast --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_168h_6h

# Attempt 2: degraded initialization and frozen backbone, seeds 2032-2036.
uv run python run_st_patchtst_ablation.py --history 24 --horizon 1 --epochs 40 --patience 8 --seeds 2032,2033,2034,2035,2036 --variants degraded_patchtst,st_station_bias_delta_forecast --initialize-from-degraded --freeze-backbone --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_frozen_24h_1h
uv run python run_st_patchtst_ablation.py --history 168 --horizon 6 --batch-size 512 --epochs 30 --patience 6 --seeds 2032,2033,2034,2035,2036 --variants degraded_patchtst,st_station_bias_delta_forecast --initialize-from-degraded --freeze-backbone --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_frozen_168h_6h

# Attempt 3: degraded initialization and 0.1x backbone LR, seeds 2037-2041.
uv run python run_st_patchtst_ablation.py --history 24 --horizon 1 --epochs 40 --patience 8 --seeds 2037,2038,2039,2040,2041 --variants degraded_patchtst,st_station_bias_delta_forecast --initialize-from-degraded --backbone-lr-scale 0.1 --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_anchored_24h_1h
uv run python run_st_patchtst_ablation.py --history 168 --horizon 6 --batch-size 512 --epochs 30 --patience 6 --seeds 2037,2038,2039,2040,2041 --variants degraded_patchtst,st_station_bias_delta_forecast --initialize-from-degraded --backbone-lr-scale 0.1 --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_anchored_168h_6h

# Attempt 4: Top-7 plus station bias with frozen backbone, seeds 2042-2046.
uv run python run_st_patchtst_ablation.py --history 24 --horizon 1 --epochs 40 --patience 8 --seeds 2042,2043,2044,2045,2046 --variants degraded_patchtst,st_sparse_station_bias_delta_forecast --initialize-from-degraded --freeze-backbone --sparse-neighbor-top-k 7 --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_sparse_station_24h_1h
uv run python run_st_patchtst_ablation.py --history 168 --horizon 6 --batch-size 512 --epochs 30 --patience 6 --seeds 2042,2043,2044,2045,2046 --variants degraded_patchtst,st_sparse_station_bias_delta_forecast --initialize-from-degraded --freeze-backbone --sparse-neighbor-top-k 7 --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_sparse_station_168h_6h

# Attempt 5: validation-selected Top-5 final confirmation, seeds 2047-2051.
uv run python run_st_patchtst_ablation.py --history 24 --horizon 1 --epochs 40 --patience 8 --seeds 2047,2048,2049,2050,2051 --variants degraded_patchtst,st_sparse_station_bias_delta_forecast --initialize-from-degraded --freeze-backbone --sparse-neighbor-top-k 5 --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_topk5_24h_1h
uv run python run_st_patchtst_ablation.py --history 168 --horizon 6 --batch-size 512 --epochs 30 --patience 6 --seeds 2047,2048,2049,2050,2051 --variants degraded_patchtst,st_sparse_station_bias_delta_forecast --initialize-from-degraded --freeze-backbone --sparse-neighbor-top-k 5 --output-dir experiments/results/st_patchtst_ablation/stability_confirmation_topk5_168h_6h

uv run python summarize_stability_confirmation.py

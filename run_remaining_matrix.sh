#!/usr/bin/env bash
# 基线对比（方向 20）剩余矩阵的顺序执行脚本（可重复运行；每个配置写完即落盘，支持续跑）。
#
# 设计要点：
#   1) 轻活优先——先把只含小模型的配置跑完，重臂（TST/ResNet/MLP）单独排在后面，
#      这样即使 GPU 中途不稳，已获得的结果覆盖最大。
#   2) 每段失败自动重试（最多 4 次，间隔 120 秒）；重试只补未完成部分。
#   3) 已知 GPU 稳定性问题：本机 amdgpu 在长时间注意力负载下会 hang 并触发模式 1 reset
#      （实测 2026-09-21 一天 16 次 reset、内核侧内存泄漏）。故：
#        - `patchtst_ci_all` 在 L=168 已在运行器内预注册排除（见工作单 §8.1b）
#        - 关闭核心转储（ulimit -c 0），避免 GB 级 core 文件
#        - 每段之间 sleep，给驱动留恢复窗口
#
# 用法：bash run_remaining_matrix.sh [--skip-guangzhou]
set -u

cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HSA_ENABLE_SDMA=0
ulimit -c 0

GRID="center_gru,center_lstm,center_tcn,plain_mix_patchtst_all,plain_mix_patchtst_top5,patchtst_ci_all,patchtst_ci_top5,multi_gru"
HEAD="center_mlp,center_resnet,center_tst,multi_tst"
LOG_DIR="${LOG_DIR:-/tmp}"
SKIP_GUANGZHOU=0
[ "${1:-}" = "--skip-guangzhou" ] && SKIP_GUANGZHOU=1

run_retry() {
  local label="$1"; shift
  local attempt=0
  until "$@"; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 4 ]; then
      echo "!!! $label 连试 4 次仍未成功，跳过该段（已落盘结果保留，可重复运行本脚本续跑）"
      return 0
    fi
    echo "!!! $label 失败，120 秒后重试（$attempt/4）$(date '+%T')"
    sleep 120
  done
}

echo "=== 段 1/5：北京 168→3/12/24（仅轻臂网格）$(date '+%F %T') ==="
run_retry 北京168轻臂 .venv/bin/python run_baseline_comparison.py --city beijing \
  --configs 168x3,168x12,168x24 \
  --grid-arms "$GRID" --headline-arms "" \
  --output-root experiments/results/baselines --device cuda
sleep 30

if [ "$SKIP_GUANGZHOU" -eq 0 ]; then
  echo "=== 段 2/5：广州矩阵（轻臂）$(date '+%F %T') ==="
  run_retry 广州 .venv/bin/python run_baseline_comparison.py --city guangzhou \
    --output-root experiments/results/baselines --device cuda
  sleep 30
fi

echo "=== 段 3/5：北京 168→6 头条（含重臂 TST/ResNet/MLP）$(date '+%F %T') ==="
run_retry 北京168头条 .venv/bin/python run_baseline_comparison.py --city beijing \
  --configs 168x6 \
  --grid-arms "$GRID" --headline-arms "$HEAD" \
  --output-root experiments/results/baselines --device cuda
sleep 30

echo "=== 段 4/5：北京 24→1 与 168→6 的重臂补齐（若前段中断）$(date '+%F %T') ==="
run_retry 北京重臂补齐 .venv/bin/python run_baseline_comparison.py --city beijing \
  --configs 24x1 \
  --grid-arms "" --headline-arms "$HEAD" \
  --output-root experiments/results/baselines --device cuda
sleep 30

echo "=== 段 5/5：端到端参考臂剩余配置 $(date '+%F %T') ==="
run_retry 参考臂 .venv/bin/python run_st_reference_arms.py \
  --output-root experiments/results/st_reference_arms --device cuda

echo "=== 全部结束 $(date '+%F %T') ==="
echo REMAINING_DONE_MARKER

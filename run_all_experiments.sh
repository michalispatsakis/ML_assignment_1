#!/usr/bin/env bash
set -euo pipefail

# Purpose: Run a standardized matrix of baseline experiments:
#   1. Baseline single GPU (FP32)
#   2. Baseline single GPU + gradient checkpointing
#   3. Baseline DDP (3 GPUs)
#   4. Baseline DDP + gradient checkpointing
# Flash / AMP variants removed per current focus. Reintroduce later if needed.

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Core hyperparameters aligned with your manual runs
MAX_STEPS=3000
LOG_INTERVAL=300
BPTT=1024
BATCH_SIZE=8          # Per-process batch size
LR=2.0                # SGD learning rate
WORLD_SIZE=3          # Number of GPUs for DDP runs

run_single() {
  DESC="$1"; shift
  LOGF="$1"; shift
  echo "============================================================"
  echo "[Single] $DESC -> $LOGF"
  echo "Command: python train_optimized.py --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} --bptt ${BPTT} --log-file ${LOGF} $@"
  echo "Started at: $(date)"
  python train_optimized.py --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} --bptt ${BPTT} --log-file ${LOGF} "$@"
  echo "Finished at: $(date)"
  echo "============================================================"
}

run_ddp() {
  DESC="$1"; shift
  LOGF="$1"; shift
  echo "============================================================"
  echo "[DDP] $DESC -> $LOGF (world_size=${WORLD_SIZE})"
  echo "Command: torchrun --nproc_per_node=${WORLD_SIZE} train_optimized.py --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} --bptt ${BPTT} --log-file ${LOGF} $@ --ddp"
  echo "Started at: $(date)"
  torchrun --nproc_per_node=${WORLD_SIZE} train_optimized.py \
    --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} --bptt ${BPTT} \
    --log-file ${LOGF} \
    --ddp "$@"
  echo "Finished at: $(date)"
  echo "============================================================"
}

###############################################
# 1) Baseline single GPU
###############################################
run_single "Baseline_Single" "baseline_single_fp32.csv" \
  --batch-size ${BATCH_SIZE} \
  --optimizer sgd --lr ${LR} --lr-schedule step

###############################################
# 2) Baseline single GPU + Gradient Checkpointing
###############################################
run_single "Baseline_Single_Checkpoint" "baseline_single_fp32_ckpt.csv" \
  --batch-size ${BATCH_SIZE} \
  --optimizer sgd --lr ${LR} --lr-schedule step \
  --grad-checkpoint

###############################################
# 3) Baseline DDP (no checkpoint)
###############################################
run_ddp "Baseline_DDP" "baseline_ddp_fp32.csv" \
  --batch-size ${BATCH_SIZE} \
  --optimizer sgd --lr ${LR} --lr-schedule step

###############################################
# 4) Baseline DDP + Gradient Checkpointing
###############################################
run_ddp "Baseline_DDP_Checkpoint" "baseline_ddp_fp32_ckpt.csv" \
  --batch-size ${BATCH_SIZE} \
  --optimizer sgd --lr ${LR} --lr-schedule step \
  --grad-checkpoint

echo "All experiments complete. Generated CSV files:"
echo "  - baseline_single_fp32.csv"
echo "  - baseline_single_fp32_ckpt.csv"
echo "  - baseline_ddp_fp32.csv"
echo "  - baseline_ddp_fp32_ckpt.csv"
echo "You can compare tokens/sec, interval_peak_mem_mb, and val metrics across these for your report."

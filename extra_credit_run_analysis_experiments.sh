#!/usr/bin/env bash
#SBATCH -J run_anal_exp_extra
#SBATCH -o ec_analysis.%j.out
#SBATCH -e ec_analysis.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -t 02:00:00

set -euo pipefail

echo "[INFO] HOSTNAME: $(hostname)"
echo "[INFO] DATE: $(date)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"

if command -v nvidia-smi &>/dev/null; then nvidia-smi || true; fi

RESULTS_DIR="results_extra_credit"
mkdir -p "${RESULTS_DIR}"

MAX_STEPS=3000
LOG_INTERVAL=300
BASELINE_BPTT=1024
BASELINE_BATCH=8
BASELINE_LR=2.0

CUDA_VISIBLE_DEVICES=0 python train_optimized.py \
  --results-dir ${RESULTS_DIR} --log-file ec_baseline_single_fp32.csv \
  --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} \
  --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 \
  --optimizer sgd --lr-schedule step &

CUDA_VISIBLE_DEVICES=1 python train_optimized.py \
  --results-dir ${RESULTS_DIR} --log-file ec_modelsize_2x_emb.csv \
  --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} \
  --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 1024 --nhead 16 --nhid 2048 --nlayers 4 --dropout 0.2 \
  --optimizer sgd --lr-schedule step &

CUDA_VISIBLE_DEVICES=2 python train_optimized.py \
  --results-dir ${RESULTS_DIR} --log-file ec_ctxlen_2048.csv \
  --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} \
  --bptt 2048 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 \
  --optimizer sgd --lr-schedule step &

wait
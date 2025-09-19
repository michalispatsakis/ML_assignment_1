#!/usr/bin/env bash
# ============================================================
# SLURM Submission Script for Analysis Experiments
# I usewd slurm to submit some of these jobs but they can be run with ./run_analysis_experiments.sh
# ============================================================
#SBATCH -J run_anal_exp                 
#SBATCH -o anal_exp_pipeline.%j.out     
#SBATCH -e anal_exp_pipeline.%j.err     
#SBATCH -p gpu-a100-small              
#SBATCH -N 1                            
#SBATCH -t 48:00:00                    

set -euo pipefail

echo "[INFO] SLURM job environment"
echo "  HOSTNAME: $(hostname)"
echo "  DATE: $(date)"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"



# Print GPU info
if command -v nvidia-smi &>/dev/null; then
  echo "[INFO] GPU Status:" && nvidia-smi || true
fi

# Runs model size scaling and context length scaling (baseline-only, no optimizations) for analysis.

RESULTS_DIR="results_analysis"
mkdir -p "${RESULTS_DIR}"

# Align with main project cadence
MAX_STEPS=3000
LOG_INTERVAL=300

# Baseline reference (kept constant unless varying explicitly):
BASELINE_BPTT=1024
BASELINE_BATCH=8
BASELINE_LR=2.0

run_cfg() {
  DESC="$1"; shift
  LOGF="$1"; shift
  echo "============================================================"
  echo "Starting: $DESC -> ${RESULTS_DIR}/${LOGF}" | tee -a "${RESULTS_DIR}/analysis_runs.log"
  echo "Command: python train_optimized.py --results-dir ${RESULTS_DIR} --log-file ${LOGF} --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} $@" | tee -a "${RESULTS_DIR}/analysis_runs.log"
  START_TS=$(date)
  python train_optimized.py --results-dir ${RESULTS_DIR} --log-file ${LOGF} --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL} "$@"
  echo "Finished: $DESC (started ${START_TS}, ended $(date))" | tee -a "${RESULTS_DIR}/analysis_runs.log"
  echo "============================================================" | tee -a "${RESULTS_DIR}/analysis_runs.log"
}

###############################################
# Model Size Scaling (3 smaller than baseline; keep context length=baseline)
# Baseline arch: emsize=512 nhead=8 nhid=1024 nlayers=4
###############################################

# 3 smaller than baseline (moderate reductions), keep context length 1024
run_cfg "MODEL_SIZE_SMALL_A" modelsize_small_a_3000.csv \
  --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 384 --nhead 6 --nhid 768 --nlayers 3 --dropout 0.2 --optimizer sgd --lr-schedule step

run_cfg "MODEL_SIZE_SMALL_B" modelsize_small_b_3000.csv \
  --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 448 --nhead 7 --nhid 896 --nlayers 3 --dropout 0.2 --optimizer sgd --lr-schedule step

run_cfg "MODEL_SIZE_2X_EMB" modelsize_2x_emb_3000.csv \
  --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 1024 --nhead 16 --nhid 2048 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step

###############################################
# Context Length Scaling (3 smaller than baseline; keep model size at baseline)
###############################################

# 2 smaller than baseline (keep training time lower)

run_cfg "CTX_LEN_512" ctxlen_512_3000.csv \
  --bptt 512 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step

run_cfg "CTX_LEN_768" ctxlen_768_3000.csv \
  --bptt 768 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step

#one large
run_cfg "CTX_LEN_2048" ctxlen_2048_3000.csv \
  --bptt 2048 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} \
  --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step

echo "All analysis experiments launched/completed. CSVs in ${RESULTS_DIR}."

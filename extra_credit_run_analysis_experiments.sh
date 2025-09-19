#!/usr/bin/env bash
#SBATCH -J run_anal_exp_extra
#SBATCH -o ec_analysis.%j.out
#SBATCH -e ec_analysis.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -t 48:00:00

set -euo pipefail

echo "[INFO] HOSTNAME: $(hostname)"
echo "[INFO] DATE: $(date)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-NA}"
echo "[INFO] CUDA_VISIBLE_DEVICES (inherited): ${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi &>/dev/null; then
  echo "[INFO] GPU Status:" && nvidia-smi || true
fi

# Use the SAME trainings as run_analysis_experiments.sh
RESULTS_DIR="results_analysis"
mkdir -p "${RESULTS_DIR}"

MAX_STEPS=3000
LOG_INTERVAL=300
BASELINE_BPTT=1024
BASELINE_BATCH=8
BASELINE_LR=2.0

SCHED_LOG="${RESULTS_DIR}/extra_credit_scheduler.log"
echo "[INFO] Scheduler started at $(date)" | tee -a "$SCHED_LOG"

# Determine GPU list (respect SLURM-provided CUDA_VISIBLE_DEVICES if set)
declare -a GPU_LIST
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
else
  # Fall back to physical IDs 0..N-1 from nvidia-smi
  if command -v nvidia-smi &>/dev/null; then
    NUM=$(nvidia-smi -L | wc -l)
  else
    NUM=1
  fi
  for ((i=0; i<NUM; i++)); do GPU_LIST+=("$i"); done
fi

NUM_GPUS=${#GPU_LIST[@]}
if (( NUM_GPUS < 1 )); then
  echo "[ERROR] No GPUs detected or visible." | tee -a "$SCHED_LOG"
  exit 1
fi
echo "[INFO] Using ${NUM_GPUS} GPU(s): ${GPU_LIST[*]}" | tee -a "$SCHED_LOG"

PYTHON=python
COMMON_PREFIX="${PYTHON} train_optimized.py --results-dir ${RESULTS_DIR} --max-steps ${MAX_STEPS} --log-interval ${LOG_INTERVAL}"

# Build task list to exactly mirror run_analysis_experiments.sh
declare -a TASKS

# Model size scaling
TASKS+=("${COMMON_PREFIX} --log-file modelsize_small_a_3000.csv --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} --emsize 384 --nhead 6 --nhid 768 --nlayers 3 --dropout 0.2 --optimizer sgd --lr-schedule step")
TASKS+=("${COMMON_PREFIX} --log-file modelsize_small_b_3000.csv --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} --emsize 448 --nhead 7 --nhid 896 --nlayers 3 --dropout 0.2 --optimizer sgd --lr-schedule step")
TASKS+=("${COMMON_PREFIX} --log-file modelsize_2x_emb_3000.csv --bptt ${BASELINE_BPTT} --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} --emsize 1024 --nhead 16 --nhid 2048 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step")

# Context length scaling
TASKS+=("${COMMON_PREFIX} --log-file ctxlen_512_3000.csv --bptt 512 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step")
TASKS+=("${COMMON_PREFIX} --log-file ctxlen_768_3000.csv --bptt 768 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step")
TASKS+=("${COMMON_PREFIX} --log-file ctxlen_2048_3000.csv --bptt 2048 --batch-size ${BASELINE_BATCH} --lr ${BASELINE_LR} --emsize 512 --nhead 8 --nhid 1024 --nlayers 4 --dropout 0.2 --optimizer sgd --lr-schedule step")

TOTAL_TASKS=${#TASKS[@]}
echo "[INFO] Queued ${TOTAL_TASKS} task(s)." | tee -a "$SCHED_LOG"

# Scheduler state
declare -A PID_TO_GPU
declare -A PID_TO_IDX
declare -a PIDS

launch_task() {
  local gpu_phys="$1"; shift
  local task_idx="$1"; shift
  local cmd="$1"; shift || true
  echo "[LAUNCH] Task #${task_idx} on GPU ${gpu_phys}: ${cmd}" | tee -a "$SCHED_LOG"
  (
    export CUDA_VISIBLE_DEVICES="${gpu_phys}"
    bash -lc "${cmd}"
  ) &
  local pid=$!
  PID_TO_GPU["$pid"]="${gpu_phys}"
  PID_TO_IDX["$pid"]="${task_idx}"
  PIDS+=("$pid")
}

# Kick off up to NUM_GPUS tasks
next_idx=0
for ((gi=0; gi<NUM_GPUS && next_idx<TOTAL_TASKS; gi++)); do
  launch_task "${GPU_LIST[$gi]}" "$next_idx" "${TASKS[$next_idx]}"
  (( next_idx++ ))
done

# Main scheduling loop: whenever any task finishes, start the next
remaining=$(( TOTAL_TASKS - next_idx ))
while (( remaining > 0 )); do
  sleep 5
  # scan for finished PIDs
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if [[ -z "$pid" ]]; then continue; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      # Reap and launch a new task on the same GPU
      wait "$pid" || echo "[WARN] Task #${PID_TO_IDX[$pid]} (pid $pid) exited with non-zero status." | tee -a "$SCHED_LOG"
      gpu_reuse="${PID_TO_GPU[$pid]}"
      echo "[DONE] Task #${PID_TO_IDX[$pid]} finished on GPU ${gpu_reuse} at $(date)." | tee -a "$SCHED_LOG"
      # Clear slot
      unset PID_TO_GPU["$pid"]
      unset PID_TO_IDX["$pid"]
      PIDS[$i]=""
      # Launch next task
      launch_task "${gpu_reuse}" "$next_idx" "${TASKS[$next_idx]}"
      (( next_idx++ ))
      (( remaining-- ))
      # Break to re-enter sleep loop giving new process time to start
      break
    fi
  done
done

# Wait for remaining running tasks
echo "[INFO] All tasks dispatched. Waiting for remaining processes..." | tee -a "$SCHED_LOG"
for pid in "${PIDS[@]}"; do
  if [[ -n "${pid}" ]]; then
    wait "$pid" || echo "[WARN] Task pid $pid exited with non-zero status." | tee -a "$SCHED_LOG"
  fi
done

echo "[INFO] Scheduler complete at $(date)." | tee -a "$SCHED_LOG"
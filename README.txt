ML Assignment 1 - How to Run

0) Install dependencies
- Ensure Python 3.9+ is available.
- Install required packages:
  pip install -r requirements.txt

1) Prepare the dataset (one-time)
- This downloads and processes WikiText-103 and writes cached tensors under ./data/
  python prepare_dataset.py

  Notes:
  - Requires internet access on first run. Subsequent runs will reuse ./data/*.pt.
  - If running on a cluster, the script sets HF caches under /home1/10917/michalakis/WORK/.cache.

2) Profiling pipeline (memory and throughput studies)
- Generate profiling CSVs:
  python profiling_experiments.py
  (Optional) Quick mode for faster turnaround:
  python profiling_experiments.py --quick

- Create plots from the profiling CSVs:
  python plot_profiling_results.py

  Outputs:
  - CSVs in profiling_results/
  - Plots in profiling_results/plots/

3) Training/optimizations (baseline + variants)
- Run the baseline matrix (single GPU, checkpointing-only, DDP-only, DDP+checkpoint):
  bash run_all_experiments.sh

  Notes:
  - DDP runs use torchrun with WORLD_SIZE=3 inside the script. Adjust WORLD_SIZE if you have a different number of GPUs.
  - Outputs go to results/ as CSV files.

4) Analysis experiments (model size and context length sweeps)
- Run the analysis sweep locally:
  bash run_analysis_experiments.sh

  Or submit via SLURM (if desired):
  sbatch run_analysis_experiments.sh

  Outputs:
  - CSVs in results_analysis/

5) Extra credit (dynamic GPU scheduler for analysis runs)
- Runs the exact same analysis trainings as run_analysis_experiments.sh but dynamically assigns the next task to any GPU that finishes first:
  bash extra_credit_run_analysis_experiments.sh

  Or submit via SLURM:
  sbatch extra_credit_run_analysis_experiments.sh

  Notes:
  - Respects CUDA_VISIBLE_DEVICES if set by SLURM; otherwise enumerates GPUs via nvidia-smi.
  - Logs scheduler activity to results_analysis/extra_credit_scheduler.log

Directory overview
- data/: cached dataset tensors (train.pt, val.pt, test.pt, vocab.pt)
- profiling_results/: profiling CSVs; plots/ contains generated PNGs
- results/: baseline and optimization CSV logs
- results_analysis/: analysis CSV logs and scheduler log

Troubleshooting
- If you hit CUDA OOM, reduce batch size (e.g., --batch-size 4) or context length (--bptt 512) in the corresponding script/command.
- For multi-GPU DDP, ensure NCCL is available and the correct number of GPUs are visible.

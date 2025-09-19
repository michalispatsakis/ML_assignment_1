import os
import csv
from typing import List, Dict
import math
import matplotlib.pyplot as plt

RESULTS_DIR = 'profiling_results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

# Remove previously generated plot files (user requested clean slate)
for fname in list(os.listdir(PLOTS_DIR)):
    if fname.endswith('.png'):
        try:
            os.remove(os.path.join(PLOTS_DIR, fname))
            print(f"[CLEAN] Deleted old plot {fname}")
        except OSError:
            pass

def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        print(f"[WARN] Missing file: {path}")
        return []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_batch_size_memory(rows: List[Dict[str, str]]):
    if not rows:
        return
    bs = [int(r['batch_size']) for r in rows]
    mem = [float(r['peak_memory_alloc_gb']) for r in rows]
    plt.figure(figsize=(5,3.2))
    plt.plot(bs, mem, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory (GB)')
    plt.title('Batch Size vs Peak Memory')
    plt.grid(alpha=0.3)
    out_path = os.path.join(PLOTS_DIR, 'batch_size_vs_memory.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved {out_path}")

def plot_batch_size_throughput(rows: List[Dict[str, str]]):
    if not rows:
        return
    bs = [int(r['batch_size']) for r in rows]
    thr = [float(r['effective_throughput_tokens_per_s']) for r in rows]
    plt.figure(figsize=(5,3.2))
    plt.plot(bs, thr, marker='s', color='tab:green')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Batch Size vs Throughput')
    plt.grid(alpha=0.3)
    out_path = os.path.join(PLOTS_DIR, 'batch_size_vs_throughput.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved {out_path}")

def plot_batch_size_throughput_amp(rows: List[Dict[str, str]]):
    if not rows:
        return
    bs = [int(r['batch_size']) for r in rows]
    thr = [float(r['effective_throughput_tokens_per_s']) for r in rows]
    plt.figure(figsize=(5,3.2))
    plt.plot(bs, thr, marker='^', color='tab:purple')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Batch Size vs Throughput (AMP)')
    plt.grid(alpha=0.3)
    out_path = os.path.join(PLOTS_DIR, 'batch_size_vs_throughput_amp.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved {out_path}")


def plot_context_length_memory(rows: List[Dict[str, str]]):
    if not rows:
        return
    ctx = [int(r['context_length']) for r in rows]
    mem = [float(r['peak_memory_alloc_gb']) for r in rows]
    plt.figure(figsize=(5,3.2))
    plt.plot(ctx, mem, marker='o')
    plt.xlabel('Context Length (bptt)')
    plt.ylabel('Peak Memory (GB)')
    plt.title('Context Length vs Peak Memory')
    plt.grid(alpha=0.3)
    out_path = os.path.join(PLOTS_DIR, 'context_length_vs_memory.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved {out_path}")


def plot_amp_memory(rows: List[Dict[str, str]]):
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: int(r['amp']))
    labels = ['FP32','AMP']
    mem = [float(r['peak_memory_alloc_gb']) for r in rows_sorted]
    plt.figure(figsize=(4.2,3.2))
    plt.bar(labels, mem, color=['tab:red','tab:red'])
    plt.ylabel('Peak Memory (GB)')
    plt.title('Precision vs Memory')
    for i,v in enumerate(mem):
        plt.text(i, v*1.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    out_path = os.path.join(PLOTS_DIR, 'precision_vs_memory.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved {out_path}")

def plot_amp_throughput(rows: List[Dict[str, str]]):
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: int(r['amp']))
    labels = ['FP32','AMP']
    thr = [float(r['effective_throughput_tokens_per_s']) for r in rows_sorted]
    plt.figure(figsize=(4.2,3.2))
    plt.bar(labels, thr, color=['tab:green','tab:green'])
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Precision vs Throughput')
    for i,v in enumerate(thr):
        plt.text(i, v*1.01, f"{v:,.0f}", ha='center', va='bottom', fontsize=8)
    out_path = os.path.join(PLOTS_DIR, 'precision_vs_throughput.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[PLOT] Saved {out_path}")


def main():
    batch_rows = read_csv(os.path.join(RESULTS_DIR, 'batch_size_vs_mem_throughput.csv'))
    batch_rows_amp = read_csv(os.path.join(RESULTS_DIR, 'batch_size_vs_mem_throughput_amp.csv'))
    ctx_rows = read_csv(os.path.join(RESULTS_DIR, 'context_length_vs_memory.csv'))
    amp_rows = read_csv(os.path.join(RESULTS_DIR, 'amp_vs_metrics.csv'))

    plot_context_length_memory(ctx_rows)          # 1) context length vs memory
    plot_batch_size_memory(batch_rows)            # 2) batch size vs memory
    plot_batch_size_throughput(batch_rows)        # 3) batch size vs throughput (FP32)
    plot_batch_size_throughput_amp(batch_rows_amp) # 4) batch size vs throughput (AMP)
    plot_amp_memory(amp_rows)                     # 5) precision (AMP) vs memory
    plot_amp_throughput(amp_rows)                 # 6) precision (AMP) vs throughput

    print('[DONE] Plots generated in', PLOTS_DIR)

if __name__ == '__main__':
    main()

import os
import time
import math
import csv
import sys
import torch
import torch.nn as nn
import argparse
from itertools import cycle
from typing import List, Dict, Optional

from model import TransformerModel
from data_preparation import get_wikitext103_data

# ---------------- Utility -----------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Throughput = tokens processed per second (data tokens * batches / elapsed)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# ---------------- Core Profiling Loop -----------------

def profile_run(batch_size: int, bptt: int, max_intervals: int, steps_per_interval: int, 
                emsize: int, nhead: int, nhid: int, nlayers: int, dropout: float,
                lr: float, amp: bool, device: torch.device, vocab, train_data, val_data,
                progress_prefix: str = "", log_interval_internal: int = 100) -> Dict[str, float]:
    """Run a short profiling segment and return aggregated metrics.
    Captures avg interval loss, avg interval val loss, mean throughput, and peak memory.
    """
    scaler = torch.amp.GradScaler('cuda', enabled=amp) if torch.cuda.is_available() else torch.cuda.amp.GradScaler(enabled=False)

    ntokens = len(vocab)
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_batches = list(range(0, train_data.size(0) - 1, bptt))
    data_iterator = cycle(train_batches)

    total_tokens = 0
    total_time = 0.0
    peak_memory_alloc = 0
    peak_memory_reserved = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    steps_total = max_intervals * steps_per_interval

    model.train()
    start_global = time.time()
    for step in range(1, steps_total + 1):
        i = next(data_iterator)
        data, targets = get_batch(train_data, i, bptt)
        data, targets = data.to(device), targets.to(device)
        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        autocast_ctx = torch.amp.autocast('cuda', enabled=amp) if torch.cuda.is_available() else torch.cuda.amp.autocast(enabled=False)
        with autocast_ctx:
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        batch_time = time.time() - t0

        # tokens processed = seq_len * batch_size
        total_tokens += data.size(0) * data.size(1)
        total_time += batch_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory_alloc = max(peak_memory_alloc, torch.cuda.max_memory_allocated(device))
            peak_memory_reserved = max(peak_memory_reserved, torch.cuda.max_memory_reserved(device))

        if step % log_interval_internal == 0 or step == steps_total:
            current_thr = (total_tokens / total_time) if total_time > 0 else 0.0
            print(f"{progress_prefix} step {step}/{steps_total} bs={batch_size} bptt={bptt} amp={amp} tokens={total_tokens} thr={current_thr:,.0f} tok/s peak_alloc={peak_memory_alloc/1024**3:.2f}GB", flush=True)

    elapsed = time.time() - start_global
    throughput_tokens_per_s = total_tokens / total_time if total_time > 0 else 0.0

    return {
        'batch_size': batch_size,
        'bptt': bptt,
        'amp': int(amp),
        'tokens_processed': total_tokens,
        'effective_throughput_tokens_per_s': throughput_tokens_per_s,
        'wall_time_s': elapsed,
        'peak_memory_alloc_gb': peak_memory_alloc / (1024**3),
        'peak_memory_reserved_gb': peak_memory_reserved / (1024**3)
    }

# ---------------- Experiment Orchestrators -----------------

def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_batch_size_vs_memory_and_throughput(device, vocab, train_data, val_data, *,
                                            batch_sizes: List[int], bptt: int,
                                            steps_per_interval: int, intervals: int,
                                            model_cfg: Dict, lr: float, amp: bool,
                                            out_dir: str, incremental: bool):
    # Use distinct filename when AMP is enabled
    csv_name = 'batch_size_vs_mem_throughput_amp.csv' if amp else 'batch_size_vs_mem_throughput.csv'
    csv_path = os.path.join(out_dir, csv_name)
    fieldnames = ['batch_size','bptt','amp','tokens_processed','effective_throughput_tokens_per_s','wall_time_s','peak_memory_alloc_gb','peak_memory_reserved_gb']
    rows = []
    for idx, bs in enumerate(batch_sizes, start=1):
        print(f"[BatchSizeExp] Starting config {idx}/{len(batch_sizes)} bs={bs}")
        # IMPORTANT: Re-batchify dataset for each batch size. Previously we reused tensors
        # already shaped with the initial base batch size (fixed number of columns), so
        # changing bs only altered a label and did not affect activation or memory usage.
        # This caused flat peak memory across batch sizes.
        fresh_train, fresh_val, _fresh_test, fresh_vocab = get_wikitext103_data(bs, bptt)
        if fresh_train.size(1) != bs:
            raise RuntimeError(f"Expected train_data width {bs}, got {fresh_train.size(1)}")
        metrics = profile_run(bs, bptt, intervals, steps_per_interval, lr=lr, amp=amp,
                               device=device, vocab=fresh_vocab, train_data=fresh_train, val_data=fresh_val,
                               progress_prefix=f"[BS {bs}] ", **model_cfg)
        rows.append(metrics)
        if incremental:
            write_csv(csv_path, fieldnames, rows)
    if not incremental:
        write_csv(csv_path, fieldnames, rows)


def run_context_length_vs_memory(device, vocab, train_data, val_data, *,
                                 ctx_lengths: List[int], batch_size: int,
                                 steps_per_interval: int, intervals: int,
                                 model_cfg: Dict, lr: float, amp: bool, out_dir: str, incremental: bool):
    csv_path = os.path.join(out_dir, 'context_length_vs_memory.csv')
    fieldnames = ['context_length','batch_size','bptt','amp','tokens_processed','effective_throughput_tokens_per_s','wall_time_s','peak_memory_alloc_gb','peak_memory_reserved_gb']
    rows = []
    for idx, ctx in enumerate(ctx_lengths, start=1):
        print(f"[CtxExp] Starting config {idx}/{len(ctx_lengths)} ctx={ctx}")
        metrics = profile_run(batch_size, ctx, intervals, steps_per_interval, lr=lr, amp=amp,
                               device=device, vocab=vocab, train_data=train_data, val_data=val_data,
                               progress_prefix=f"[CTX {ctx}] ", **model_cfg)
        rows.append(metrics | {'context_length': ctx})
        if incremental:
            write_csv(csv_path, fieldnames, rows)
    if not incremental:
        write_csv(csv_path, fieldnames, rows)


def run_amp_vs_metrics(device, vocab, train_data, val_data, *,
                       batch_size: int, bptt: int, steps_per_interval: int, intervals: int,
                       model_cfg: Dict, lr: float, out_dir: str, incremental: bool):
    csv_path = os.path.join(out_dir, 'amp_vs_metrics.csv')
    fieldnames = ['batch_size','bptt','amp','tokens_processed','effective_throughput_tokens_per_s','wall_time_s','peak_memory_alloc_gb','peak_memory_reserved_gb']
    rows = []
    for idx, amp_flag in enumerate([False, True], start=1):
        print(f"[AMPExp] Starting config {idx}/2 amp={amp_flag}")
        metrics = profile_run(batch_size, bptt, intervals, steps_per_interval, lr=lr, amp=amp_flag,
                               device=device, vocab=vocab, train_data=train_data, val_data=val_data,
                               progress_prefix=f"[AMP {amp_flag}] ", **model_cfg)
        rows.append(metrics)
        if incremental:
            write_csv(csv_path, fieldnames, rows)
    if not incremental:
        write_csv(csv_path, fieldnames, rows)

# ---------------- Main Entry -----------------

def parse_args():
    p = argparse.ArgumentParser(description='Profiling Experiments')
    p.add_argument('--quick', action='store_true', help='Run in quick mode (fewer steps)')
    p.add_argument('--out-dir', default='profiling_results', help='Output directory')
    p.add_argument('--incremental', action='store_true', help='Write CSV after each config')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Match baseline model and optimizer hyperparameters used in training
    model_cfg = dict(emsize=512, nhead=8, nhid=1024, nlayers=4, dropout=0.2)
    lr = 2.0  # Baseline SGD learning rate

    base_batch_size = 8   # Baseline per-process batch size
    base_bptt = 1024      # Baseline context length (sequence length)
    print('[Data] Loading cached dataset tensors...')
    train_data, val_data, test_data, vocab = get_wikitext103_data(base_batch_size, base_bptt)

    # Default to QUICK-like settings (one interval) as we only need a single steady-state snapshot
    if args.quick:
        steps_per_interval = 50
        intervals = 1
        print('[Mode] QUICK: steps_per_interval=50 intervals=1 (50 steps/config)')
    else:
        steps_per_interval = 100
        intervals = 1
        print('[Mode] DEFAULT: steps_per_interval=100 intervals=1 (100 steps/config)')

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('[Experiment] Batch size vs memory & throughput (baseline params, vary batch size only)')
    run_batch_size_vs_memory_and_throughput(
        device, vocab, train_data, val_data,
        batch_sizes=[8, 16, 32, 48], bptt=base_bptt,
        steps_per_interval=steps_per_interval, intervals=intervals,
        model_cfg=model_cfg, lr=lr, amp=False, out_dir=out_dir, incremental=args.incremental
    )

    print('[Experiment] Batch size vs memory & throughput with AMP (vary batch size only)')
    # Run a mirrored sweep with AMP enabled and write a separate CSV
    # This allows separate plotting for FP32 vs AMP throughput curves by batch size.
    # We will reuse the same function and simply toggle amp=True.
    run_batch_size_vs_memory_and_throughput(
        device, vocab, train_data, val_data,
        batch_sizes=[8, 16, 32, 48], bptt=base_bptt,
        steps_per_interval=steps_per_interval, intervals=intervals,
        model_cfg=model_cfg, lr=lr, amp=True, out_dir=out_dir, incremental=args.incremental
    )

    print('[Experiment] Context length vs memory (baseline params, vary context length only)')
    run_context_length_vs_memory(
        device, vocab, train_data, val_data,
        # Two smaller and two larger than baseline 1024
        ctx_lengths=[512, 768, 1536, 2048], batch_size=base_batch_size,
        steps_per_interval=steps_per_interval, intervals=intervals,
        model_cfg=model_cfg, lr=lr, amp=False, out_dir=out_dir, incremental=args.incremental
    )

    print('[Experiment] AMP on/off vs metrics (baseline params, toggle AMP only)')
    run_amp_vs_metrics(
        device, vocab, train_data, val_data,
        batch_size=base_batch_size, bptt=base_bptt,
        steps_per_interval=steps_per_interval, intervals=intervals,
        model_cfg=model_cfg, lr=lr, out_dir=out_dir, incremental=args.incremental
    )

    print('Profiling complete. CSV files written to', out_dir)

if __name__ == '__main__':
    main()

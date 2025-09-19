import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import cycle
from data_preparation import get_wikitext103_data
from model import TransformerModel
from profiling import CSVLogger
from train import get_batch

def parse_args():
    p = argparse.ArgumentParser(description='Optimized training with optional Flash Attention + AMP + optional DDP')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--bptt', type=int, default=1024)
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--log-interval', type=int, default=500)
    p.add_argument('--lr', type=float, default=5.0)
    p.add_argument('--min-lr', type=float, default=1e-5, help='Minimum LR floor for cosine schedule / warmup floor')
    p.add_argument('--warmup-steps', type=int, default=400, help='Linear warmup steps (0 to disable)')
    p.add_argument('--results-dir', type=str, default='results')
    p.add_argument('--log-file', type=str, default='train_optimized.csv')
    # Model size configuration (for analysis sweeps)
    p.add_argument('--emsize', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--nhid', type=int, default=1024)
    p.add_argument('--nlayers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--flash-attn', action='store_true', help='Enable flash attention kernels (causal scaled dot product)')
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (autocast + GradScaler)')
    p.add_argument('--ddp', action='store_true', help='Enable Distributed Data Parallel (multi-GPU)')
    p.add_argument('--dist-backend', type=str, default='nccl', help='Preferred distributed backend (nccl or gloo)')
    p.add_argument('--dist-url', type=str, default='env://', help='Use torchrun with env:// init')
    p.add_argument('--optimizer', type=str, default='sgd', choices=['sgd','adamw'])
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--momentum', type=float, default=0.0, help='Momentum (SGD only)')
    p.add_argument('--lr-schedule', type=str, default='step', choices=['step','cosine'])
    p.add_argument('--grad-checkpoint', action='store_true', help='Enable gradient checkpointing per encoder layer (reduces activation memory)')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, args.log_file)
    # Only rank 0 (or single process) clears existing log.
    if (not args.ddp) or int(os.environ.get('RANK', '0')) == 0:
        if os.path.exists(log_path):
            try:
                os.remove(log_path)
                print(f'Removed existing log file: {log_path}')
            except FileNotFoundError:
                pass

    emsize = args.emsize
    nhead = args.nhead
    nhid = args.nhid
    nlayers = args.nlayers
    dropout = args.dropout

    # DDP init (lazy)
    def init_distributed(preferred_backend: str):
        chosen = preferred_backend
        if preferred_backend == 'nccl':
            try:
                if args.dist_url == 'env://':
                    dist.init_process_group(backend='nccl')
                else:
                    dist.init_process_group(backend='nccl', init_method=args.dist_url)
                return 'nccl'
            except RuntimeError as e:
                if 'NCCL' in str(e).upper():
                    print('[WARN] NCCL backend unavailable; falling back to gloo.')
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    if args.dist_url == 'env://':
                        dist.init_process_group(backend='gloo')
                    else:
                        dist.init_process_group(backend='gloo', init_method=args.dist_url)
                    chosen = 'gloo'
                else:
                    raise
        else:
            if args.dist_url == 'env://':
                dist.init_process_group(backend=preferred_backend)
            else:
                dist.init_process_group(backend=preferred_backend, init_method=args.dist_url)
        return chosen

    if args.ddp:
        backend_used = init_distributed(args.dist_backend)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if rank == 0:
            print(f'[INFO] Distributed initialized with backend={backend_used}, world_size={world_size}, local_rank={local_rank}')
    else:
        local_rank = 0
        world_size = 1
        rank = 0

    train_data, val_data, test_data, vocab = get_wikitext103_data(args.batch_size, args.bptt)
    ntokens = len(vocab)
    model = TransformerModel(
        ntoken=ntokens, ninp=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout,
        use_checkpoint=args.grad_checkpoint, use_flash_attn=args.flash_attn
    )
    device = torch.device('cuda', local_rank) if (args.ddp and torch.cuda.is_available()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.ddp:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == 'cuda' else None,
            output_device=device.index if device.type == 'cuda' else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adamw':
        if args.lr > 0.01:
            if rank == 0:
                print(f'[WARN] AdamW lr={args.lr} high; auto-reducing to 0.001.')
            args.lr = 0.001
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    base_lr = args.lr
    if args.lr_schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    else:
        scheduler = None

    logger = None
    if rank == 0:
        logger = CSVLogger(log_path, headers=['step','lr','ms_per_batch','global_tokens_sec','train_loss','train_ppl','train_acc','val_loss','val_ppl','val_acc','alloc_mem_mb','interval_peak_mem_mb','cumulative_peak_mem_mb','elapsed_time_s','amp','ddp','param_count'])

    best_val_loss = float('inf')
    total_steps = 0
    total_loss = 0.0
    total_correct_interval = 0
    total_tokens_interval = 0

    # Build index list and shard across ranks if DDP
    train_batches_all = list(range(0, train_data.size(0) - 1, args.bptt))
    if args.ddp:
        train_batches = train_batches_all[rank::world_size]
        if len(train_batches) == 0:
            train_batches = train_batches_all
            if rank == 0:
                print('[WARN] Dataset too small for sharding; all ranks use full set.')
    else:
        train_batches = train_batches_all
    data_iterator = cycle(train_batches)

    if rank == 0:
        param_count = model.module.trainable_parameter_count if isinstance(model, DDP) else model.trainable_parameter_count
        print(f'Starting training (flash_attn={args.flash_attn}, ddp={args.ddp}, amp={args.amp})')
        print(f'Parameters: {param_count:,}')

    start_time = time.time()
    global_start = start_time
    cumulative_peak_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    model.train()

    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
        autocast_ctx = autocast
    except ImportError:
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
        autocast_ctx = torch.cuda.amp.autocast

    while total_steps < args.max_steps:
        i = next(data_iterator)
        data, targets = get_batch(train_data, i, args.bptt)
        data, targets = data.to(device), targets.to(device)
        if args.flash_attn:
            src_mask = None
        else:
            if isinstance(model, DDP):
                src_mask = model.module.generate_square_subsequent_mask(data.size(0)).to(device)
            else:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device_type='cuda', enabled=args.amp and torch.cuda.is_available()):
            output = model(data, src_mask)
            logits = output.view(-1, ntokens)
            loss = criterion(logits, targets)
        if args.amp and torch.cuda.is_available():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if args.amp and torch.cuda.is_available():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct_interval += (preds == targets).sum().item()
            total_tokens_interval += targets.numel()
        total_steps += 1

        if total_steps % args.log_interval == 0:
            # Warmup LR adjustment
            if args.warmup_steps > 0 and total_steps <= args.warmup_steps:
                warmup_lr = args.lr * (total_steps / args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = max(warmup_lr, args.min_lr)
            lr_current = optimizer.param_groups[0]['lr']
            interval_elapsed = time.time() - start_time
            ms_per_batch = (interval_elapsed * 1000) / args.log_interval
            cur_loss = total_loss / args.log_interval
            try:
                ppl = math.exp(cur_loss)
            except OverflowError:
                ppl = float('inf')
            train_acc = (total_correct_interval / total_tokens_interval) if total_tokens_interval>0 else 0.0
            global_tokens_interval = total_tokens_interval * (world_size if args.ddp else 1)
            tokens_per_sec = global_tokens_interval / interval_elapsed if interval_elapsed>0 else 0.0

            def eval_loop(eval_data):
                model_was_training = model.training
                model.eval()
                loss_sum_tokens = 0.0
                correct_sum = 0.0
                token_count = 0.0
                with torch.no_grad():
                    for j in range(0, eval_data.size(0) - 1, args.bptt):
                        d, t = get_batch(eval_data, j, args.bptt)
                        d, t = d.to(device), t.to(device)
                        if args.flash_attn:
                            out = model(d, None)
                        else:
                            mask = model.module.generate_square_subsequent_mask(d.size(0)).to(device) if isinstance(model, DDP) else model.generate_square_subsequent_mask(d.size(0)).to(device)
                            out = model(d, mask)
                        le = criterion(out.view(-1, ntokens), t)
                        tokens = t.numel()
                        loss_sum_tokens += le.item() * tokens
                        preds_eval = out.view(-1, ntokens).argmax(dim=-1)
                        correct_sum += (preds_eval == t).sum().item()
                        token_count += tokens
                if args.ddp:
                    metrics = torch.tensor([loss_sum_tokens, correct_sum, token_count], dtype=torch.float64, device=device)
                    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                    loss_sum_tokens, correct_sum, token_count = metrics.tolist()
                avg_loss_eval = (loss_sum_tokens / token_count) if token_count>0 else 0.0
                acc_eval = (correct_sum / token_count) if token_count>0 else 0.0
                if model_was_training:
                    model.train()
                return avg_loss_eval, acc_eval

            val_loss, val_acc = eval_loop(val_data)
            try:
                val_ppl = math.exp(val_loss)
            except OverflowError:
                val_ppl = float('inf')

            if torch.cuda.is_available():
                alloc_bytes = torch.cuda.memory_allocated(device)
                interval_peak_bytes = torch.cuda.max_memory_allocated(device)
                cumulative_peak_bytes = max(cumulative_peak_bytes, interval_peak_bytes)
                alloc_mb = alloc_bytes / (1024**2)
                interval_peak_mb = interval_peak_bytes / (1024**2)
                cumulative_peak_mb = cumulative_peak_bytes / (1024**2)
            else:
                alloc_mb = 0.0
                interval_peak_mb = 0.0
                cumulative_peak_mb = 0.0
            elapsed_time_s = time.time() - global_start
            if rank == 0:
                print(
                    f'| step {total_steps:5d}/{args.max_steps:5d} | lr {lr_current:8.6f} | ms/batch {ms_per_batch:7.2f} | '
                    f'tok/s {tokens_per_sec/1000:7.2f}k | loss {cur_loss:5.2f} | ppl {ppl:8.2f} | train_acc {train_acc*100:5.2f}% | '
                    f'val_loss {val_loss:5.2f} | val_ppl {val_ppl:8.2f} | val_acc {val_acc*100:5.2f}% | '
                    f'alloc {alloc_mb:7.1f}MB | int_peak {interval_peak_mb:7.1f}MB | cum_peak {cumulative_peak_mb:7.1f}MB | elapsed {elapsed_time_s:8.1f}s | '
                    f'amp {int(args.amp)} | ddp {int(args.ddp)}'
                )
                logger.log([
                    total_steps, lr_current, ms_per_batch, tokens_per_sec, cur_loss, ppl, train_acc, val_loss, val_ppl, val_acc,
                    alloc_mb, interval_peak_mb, cumulative_peak_mb, elapsed_time_s, int(args.amp), int(args.ddp), int(param_count)
                ])
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    torch.save(state, os.path.join(args.results_dir, 'best_model_best.pth'))

            if not (args.warmup_steps > 0 and total_steps <= args.warmup_steps):
                if args.lr_schedule == 'step' and scheduler is not None:
                    scheduler.step()
                elif args.lr_schedule == 'cosine':
                    progress = (total_steps - args.warmup_steps) / max(1, (args.max_steps - args.warmup_steps))
                    progress = min(max(progress, 0.0), 1.0)
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                    target_lr = args.min_lr + (base_lr - args.min_lr) * cosine_factor
                    for pg in optimizer.param_groups:
                        pg['lr'] = target_lr

            total_loss = 0.0
            total_correct_interval = 0
            total_tokens_interval = 0
            start_time = time.time()
            model.train()

    if os.path.exists(os.path.join(args.results_dir, 'best_model_best.pth')):
        if (not args.ddp) or dist.get_rank() == 0:
            loaded = torch.load(os.path.join(args.results_dir, 'best_model_best.pth'), map_location=device)
            if isinstance(model, DDP):
                model.module.load_state_dict(loaded)
            else:
                model.load_state_dict(loaded)

    def final_eval(eval_data):
        model_was_training = model.training
        model.eval()
        loss_sum_tokens = 0.0
        correct_sum = 0.0
        token_count = 0.0
        with torch.no_grad():
            for j in range(0, eval_data.size(0) - 1, args.bptt):
                d, t = get_batch(eval_data, j, args.bptt)
                d, t = d.to(device), t.to(device)
                if args.flash_attn:
                    out = model(d, None)
                else:
                    mask = model.module.generate_square_subsequent_mask(d.size(0)).to(device) if isinstance(model, DDP) else model.generate_square_subsequent_mask(d.size(0)).to(device)
                    out = model(d, mask)
                logit_eval = out.view(-1, ntokens)
                loss_eval = criterion(logit_eval, t)
                tokens = t.numel()
                loss_sum_tokens += loss_eval.item() * tokens
                preds_eval = logit_eval.argmax(dim=-1)
                correct_sum += (preds_eval == t).sum().item()
                token_count += tokens
        if args.ddp:
            metrics = torch.tensor([loss_sum_tokens, correct_sum, token_count], dtype=torch.float64, device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            loss_sum_tokens, correct_sum, token_count = metrics.tolist()
        avg_loss_eval = (loss_sum_tokens / token_count) if token_count>0 else 0.0
        acc_eval = (correct_sum / token_count) if token_count>0 else 0.0
        if model_was_training:
            model.train()
        return avg_loss_eval, acc_eval

    test_loss, test_acc = final_eval(test_data)
    test_ppl = math.exp(test_loss)
    total_elapsed = time.time() - global_start
    if (not args.ddp) or dist.get_rank() == 0:
        print('=' * 110)
        print(f'| End (optimized) | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f} | test_acc {test_acc*100:5.2f}% | total_time {total_elapsed:8.1f}s | amp {int(args.amp)} | ddp {int(args.ddp)} | flash_attn {int(args.flash_attn)}')
        print('=' * 110)

    if args.ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import math
import time
import os
from itertools import cycle
from profiling import CSVLogger

def token_accuracy(logits, targets):
    """Compute token-level accuracy given model output (T,B,V) reshaped to (T*B,V) vs targets (T*B)."""
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        return correct / targets.numel()

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def evaluate(model, eval_data, criterion, ntokens, bptt, device):
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            data, targets = data.to(device), targets.to(device)
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = model(data, src_mask) # (seq_len, batch, vocab)
            logits = output.view(-1, ntokens)
            loss = criterion(logits, targets)
            total_loss += len(data) * loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
    avg_loss = total_loss / (len(eval_data) - 1)
    acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, acc

def train_model(model, train_data, val_data, test_data, vocab, max_steps, log_interval, results_dir, bptt, lr, log_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Align CSV format with results_dir (from train_optimized.py)
    # Includes throughput and both interval and cumulative peak memory metrics.
    logger = CSVLogger(log_file, headers=[
        'step','lr','ms_per_batch','global_tokens_sec',
        'train_loss','train_ppl','train_acc','val_loss','val_ppl','val_acc',
        'alloc_mem_mb','interval_peak_mem_mb','cumulative_peak_mem_mb','elapsed_time_s','amp','ddp'
    ])

    best_val_loss = float('inf')
    total_steps = 0
    
    ntokens = len(vocab)
    
    print("Starting training...")
    
    model.train()
    total_loss = 0.
    total_correct_interval = 0
    total_tokens_interval = 0
    start_time = time.time()
    global_start = start_time
    
    # Use itertools.cycle to loop over the data indefinitely
    train_batches = list(range(0, train_data.size(0) - 1, bptt))
    data_iterator = cycle(train_batches)

    # Track cumulative peak across intervals
    cumulative_peak_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    while total_steps < max_steps:
        i = next(data_iterator)
        
        data, targets = get_batch(train_data, i, bptt)
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        logits = output.view(-1, ntokens)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        # accumulate training accuracy for interval
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct_interval += (preds == targets).sum().item()
            total_tokens_interval += targets.numel()
        total_steps += 1

        if total_steps % log_interval == 0 and total_steps > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            train_acc = (total_correct_interval / total_tokens_interval) if total_tokens_interval>0 else 0.0
            interval_elapsed = (time.time() - start_time)
            tokens_per_sec = (total_tokens_interval / interval_elapsed) if interval_elapsed > 0 else 0.0

            # Validation
            val_loss, val_acc = evaluate(model, val_data, criterion, ntokens, bptt, device)
            val_ppl = math.exp(val_loss)

            # Memory (only meaningful if CUDA). Reset peak after logging to measure interval peaks if desired.
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
            print(f'| step {total_steps:5d}/{max_steps:5d} | lr {lr:4.2f} | '
                  f'ms/batch {ms_per_batch:5.2f} | tok/s {tokens_per_sec/1000:7.2f}k | loss {cur_loss:5.2f} | ppl {ppl:8.2f} | '
                  f'train_acc {train_acc*100:5.2f}% | val_loss {val_loss:5.2f} | val_ppl {val_ppl:8.2f} | val_acc {val_acc*100:5.2f}% | '
                  f'alloc {alloc_mb:7.1f}MB | int_peak {interval_peak_mb:7.1f}MB | cum_peak {cumulative_peak_mb:7.1f}MB | elapsed {elapsed_time_s:8.1f}s')

            logger.log([
                total_steps, lr, ms_per_batch, tokens_per_sec,
                cur_loss, ppl, train_acc, val_loss, val_ppl, val_acc,
                alloc_mb, interval_peak_mb, cumulative_peak_mb, elapsed_time_s, 0, 0
            ])

            # Reset peak stats so next interval peak is isolated
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

            scheduler.step()

            # Reset for next logging interval
            total_loss = 0
            total_correct_interval = 0
            total_tokens_interval = 0
            start_time = time.time()
            model.train() # Switch back to train mode

    # Final evaluation on test set
    if os.path.exists(os.path.join(results_dir, 'best_model.pth')):
        model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    test_loss, test_acc = evaluate(model, test_data, criterion, ntokens, bptt, device)
    test_ppl = math.exp(test_loss)
    total_elapsed = time.time() - global_start
    print('=' * 110)
    print(f'| End of training | test loss {test_loss:5.2f} | test ppl {test_ppl:8.2f} | test_acc {test_acc*100:5.2f}% | total_time {total_elapsed:8.1f}s')
    print('=' * 110)



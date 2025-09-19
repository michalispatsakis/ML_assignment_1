import os
import math
import argparse
import torch
import torch.nn as nn
from data_preparation import get_wikitext103_data
from model import TransformerModel
from train import train_model  # Reuses existing training loop (non-DDP / baseline)


def parse_args():
    p = argparse.ArgumentParser(description='Baseline Transformer training with configurable size and context length')
    p.add_argument('--results-dir', type=str, default='results_analysis', help='Directory to store result CSVs')
    p.add_argument('--log-file', type=str, default='baseline_config.csv', help='CSV log filename')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--bptt', type=int, default=1024, help='Context length (sequence length)')
    p.add_argument('--emsize', type=int, default=512, help='Embedding (model) dimension')
    p.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    p.add_argument('--nhid', type=int, default=1024, help='Feedforward hidden dimension')
    p.add_argument('--nlayers', type=int, default=4, help='Number of Transformer encoder layers')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=5.0)
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--log-interval', type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, args.log_file)
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Removed existing log file: {log_path}")

    print("Config:")
    print(f"  emsize={args.emsize} nhead={args.nhead} nhid={args.nhid} nlayers={args.nlayers} bptt={args.bptt} batch_size={args.batch_size}")

    # Load data (reshaped according to batch size & bptt inside helper)
    train_data, val_data, test_data, vocab = get_wikitext103_data(args.batch_size, args.bptt)
    ntokens = len(vocab)

    model = TransformerModel(ntoken=ntokens,
                             ninp=args.emsize,
                             nhead=args.nhead,
                             nhid=args.nhid,
                             nlayers=args.nlayers,
                             dropout=args.dropout,
                             use_checkpoint=False,
                             use_flash_attn=False)

    # Reuse baseline training utility (single GPU / CPU)
    train_model(model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                vocab=vocab,
                max_steps=args.max_steps,
                log_interval=args.log_interval,
                results_dir=args.results_dir,
                bptt=args.bptt,
                lr=args.lr,
                log_file=args.log_file)


if __name__ == '__main__':
    main()

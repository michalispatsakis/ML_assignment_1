import os
import sys
import torch
from torch.utils.data import Dataset
from vocab import Vocab
import math

def batchify(data, batch_size):
    # Work with 1-D LongTensor, reshape into batch_size columns
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data

def load_vocab(vocab_path: str) -> Vocab:
    """Robustly load vocabulary supporting multiple historical serialization formats.

    Formats:
      1. dict {'itos': [...]} (current)
      2. dict {'stoi': {...}, 'itos': [...]} (earlier state dict)
      3. Pickled Vocab instance referencing __main__.Vocab
    """
    def build(tokens):
        v = object.__new__(Vocab)
        v.itos = tokens
        v.stoi = {tok: i for i, tok in enumerate(tokens)}
        return v

    try:
        payload = torch.load(vocab_path, weights_only=False)
    except AttributeError:
        main_mod = sys.modules.get('__main__')
        if main_mod is not None and not hasattr(main_mod, 'Vocab'):
            setattr(main_mod, 'Vocab', Vocab)
        payload = torch.load(vocab_path, weights_only=False)

    if isinstance(payload, dict):
        if 'itos' in payload:
            return build(payload['itos'])
    if isinstance(payload, Vocab):
        return payload
    raise RuntimeError(f"Unrecognized vocab serialization format: {type(payload)}")


def get_wikitext103_data(batch_size, bptt):
    """Loads pre-processed WikiText-103 data and vocabulary from the 'data' directory."""
    data_dir = 'data'
    vocab = load_vocab(os.path.join(data_dir, 'vocab.pt'))

    def safe_load(path):
        try:
            return torch.load(path, weights_only=True)
        except TypeError:
            # Older PyTorch without weights_only param
            return torch.load(path)

    train_data = safe_load(os.path.join(data_dir, 'train.pt'))
    val_data = safe_load(os.path.join(data_dir, 'val.pt'))
    test_data = safe_load(os.path.join(data_dir, 'test.pt'))

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)
    test_data = batchify(test_data, batch_size)
    
    return train_data, val_data, test_data, vocab


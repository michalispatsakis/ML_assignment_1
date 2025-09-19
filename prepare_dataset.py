import os
import torch
from datasets import load_dataset
from collections import Counter
from vocab import Vocab

def prepare_dataset():
    """
    Downloads, processes, and saves the WikiText-103 dataset.
    """
    # Set environment variables for Hugging Face
    os.environ['HF_HOME'] = '/home1/10917/michalakis/WORK/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/home1/10917/michalakis/WORK/.cache/hf_datasets'
    
    print("Loading WikiText-103 dataset from Hugging Face...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')

    print("Building vocabulary...")
    counter = Counter()
    for item in dataset['train']:
        counter.update(item['text'].split())
    
    vocab = Vocab(counter)
    print(f"Vocabulary size: {len(vocab)}")

    def data_process(raw_text_iterator):
        data = [torch.tensor([vocab.stoi.get(token, vocab.stoi['<unk>']) for token in item['text'].split()], dtype=torch.long) for item in raw_text_iterator if item['text'].strip() != '']
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    train_data = data_process(dataset['train'])
    val_data = data_process(dataset['validation'])
    test_data = data_process(dataset['test'])

    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    torch.save(train_data, os.path.join(data_dir, 'train.pt'))
    torch.save(val_data, os.path.join(data_dir, 'val.pt'))
    torch.save(test_data, os.path.join(data_dir, 'test.pt'))
    # Save only the ordered token list; reconstruction will rebuild stoi
    torch.save({'itos': vocab.itos}, os.path.join(data_dir, 'vocab.pt'))

    print("Dataset preparation complete. Processed data saved to the 'data' directory.")

if __name__ == '__main__':
    prepare_dataset()

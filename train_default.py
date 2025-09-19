import os
from data_preparation import get_wikitext103_data
from model import TransformerModel
from train import train_model
from profiling import CSVLogger

def main():
    # Create results directory and clear previous results
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    best_model_path = os.path.join(results_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        print(f"Removed existing model file: {best_model_path}")

    log_file = os.path.join(results_dir, 'train_default_parameters.csv')

    # Hyperparameters
    batch_size = 16
    eval_batch_size = 8
    bptt = 1024
    ntoken = 0  # Will be updated after loading data
    emsize = 512
    nhead = 8
    nhid = 1024
    nlayers = 4
    dropout = 0.2
    lr = 5.0
    max_steps = 5000
    log_interval = 500

    # Load data
    train_data, val_data, test_data, vocab = get_wikitext103_data(batch_size, bptt)
    ntoken = len(vocab)

    # Initialize model
    model = TransformerModel(ntoken, emsize, nhead, nhid, nlayers, dropout)

    # Train model
    train_model(model, train_data, val_data, test_data, vocab, max_steps, log_interval, results_dir, bptt, lr, log_file)

if __name__ == '__main__':
    main()

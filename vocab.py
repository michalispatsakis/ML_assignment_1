class Vocab:
    """A vocabulary class to map tokens to indices."""
    def __init__(self, counter, max_size=30000):
        self.itos = ["<unk>", "<pad>"]
        self.itos.extend([token for token, _ in counter.most_common(max_size - 2)])
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

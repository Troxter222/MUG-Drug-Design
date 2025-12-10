class Vocabulary:
    def __init__(self, vocab_list):
        self.vocab = vocab_list
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        
    def __len__(self):
        return len(self.vocab)

    def decode(self, indices):
        tokens = []
        for i in indices:
            token = self.idx2char[i]

            if token == '<eos>': 
                break
            if token != '<pad>' and token != '<sos>':
                tokens.append(token)
                
        return "".join(tokens)
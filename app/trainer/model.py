"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import selfies as sf


class Vocabulary:
    def __init__(self, vocab_list):
        self.vocab = vocab_list
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def encode(self, selfie_str, max_len=100):
        try:
            tokens = list(sf.split_selfies(selfie_str))
        except Exception:
            tokens = []

        indices = [self.char2idx['<sos>']] + \
                  [self.char2idx.get(t, self.char2idx['<pad>']) for t in tokens] + \
                  [self.char2idx['<eos>']]

        if len(indices) < max_len:
            indices += [self.char2idx['<pad>']] * (max_len - len(indices))
        return indices[:max_len]

    def decode(self, indices):
        tokens = []
        for i in indices:
            token = self.idx2char[i]
            if token == '<eos>':
                break
            if token != '<pad>' and token != '<sos>':
                tokens.append(token)

        return "".join(tokens)


# --- NEURAL NETWORK ---
class MolecularVAE(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=256,
                 latent_size=128, num_layers=3):
        super(MolecularVAE, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.encoder_gru = nn.GRU(
            embed_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=0.1
        )
        self.fc_mu = nn.Linear(hidden_size * num_layers, latent_size)
        self.fc_logvar = nn.Linear(hidden_size * num_layers, latent_size)

        self.decoder_input = nn.Linear(latent_size, hidden_size * num_layers)
        self.decoder_gru = nn.GRU(
            embed_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=0.1
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):
        embedded = self.embedding(x)
        _, hidden = self.encoder_gru(embedded)
        # Flatten hidden state to match linear layer input
        hidden = hidden.permute(1, 0, 2).contiguous().view(hidden.size(1), -1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        hidden = self.decoder_input(z)
        hidden = hidden.view(
            x.size(0), self.num_layers, self.hidden_size
        ).permute(1, 0, 2).contiguous()

        embedded = self.embedding(x)
        output, _ = self.decoder_gru(embedded, hidden)
        prediction = self.fc_out(output)
        return prediction, mu, logvar

    def sample(self, n_samples, device, vocab, max_len=100, temp=1.0):
        z = torch.randn(n_samples, self.latent_size).to(device)

        hidden = self.decoder_input(z)
        hidden = hidden.view(
            n_samples, self.num_layers, self.hidden_size
        ).permute(1, 0, 2).contiguous()

        input_idx = torch.tensor(
            [[vocab.char2idx['<sos>']]] * n_samples
        ).to(device)
        decoded_indices = []

        for _ in range(max_len):
            embedded = self.embedding(input_idx)
            output, hidden = self.decoder_gru(embedded, hidden)
            logits = self.fc_out(output)

            probs = F.softmax(logits.squeeze(1) / temp, dim=-1)
            next_token = torch.multinomial(probs, 1)

            decoded_indices.append(next_token)
            input_idx = next_token

        return torch.cat(decoded_indices, dim=1)
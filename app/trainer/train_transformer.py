"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import json
import os
import torch
import torch.nn as nn
import pandas as pd
import selfies as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from app.core.transformer_model import MoleculeTransformer


class Config:
    """Configuration parameters for Transformer training."""
    FILE_DATA = 'dataset/processed_v2/transformer_train.csv'
    FILE_VOCAB = 'dataset/processed_v2/vocab_transformer.json'
    SAVE_DIR = 'checkpoints_transformer_v3'

    BATCH_SIZE = 64
    EPOCHS = 30
    MAX_LEN = 120

    D_MODEL = 256
    NHEAD = 8
    LAYERS = 4
    LATENT = 128

    WARMUP_STEPS = 4000

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfiesDataset(Dataset):
    def __init__(self, csv_file, vocab_file, max_len):
        self.df = pd.read_csv(csv_file)
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.max_len = max_len
        self.pad_idx = self.char2idx['<pad>']
        self.sos_idx = self.char2idx['<sos>']
        self.eos_idx = self.char2idx['<eos>']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        selfies_str = str(self.df.iloc[idx]['selfies'])
        try:
            tokens = list(sf.split_selfies(selfies_str))
        except Exception:
            tokens = []

        # Add SOS and EOS tokens
        indices = [self.sos_idx] + \
                  [self.char2idx.get(t, self.pad_idx) for t in tokens] + \
                  [self.eos_idx]

        # Padding
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        indices += [self.pad_idx] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long)


class NoamOpt:
    """Optim wrapper that implements the Noam learning rate schedule."""
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


def generate_square_subsequent_mask(sz):
    """
    Generates a causal mask to ensure the decoder does not see future tokens.
    Returns a matrix where upper triangle is -inf and rest is 0.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')) \
        .masked_fill(mask == 1, float(0.0))
    return mask


def train():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    print(f"TRANSFORMER V2 (Corrected Logic) START on {Config.DEVICE}")

    dataset = SelfiesDataset(
        Config.FILE_DATA, Config.FILE_VOCAB, Config.MAX_LEN
    )
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    vocab_size = len(dataset.vocab)

    model = MoleculeTransformer(
        vocab_size, Config.D_MODEL, Config.NHEAD,
        Config.LAYERS, Config.LAYERS,
        Config.D_MODEL * 4, Config.LATENT
    ).to(Config.DEVICE)

    optimizer = NoamOpt(
        Config.D_MODEL, 1, Config.WARMUP_STEPS,
        torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
    )

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0

        progress = tqdm(dataloader, desc=f"Ep {epoch + 1}")

        for batch in progress:
            # batch: [Batch, Seq] -> Transpose -> [Seq, Batch]
            # PyTorch Transformer expects [Seq, Batch, Emb] by default
            batch = batch.transpose(0, 1).to(Config.DEVICE)

            # 1. Encoder input: The FULL sequence
            # (Allows the encoder to understand the entire molecule context)
            encoder_input = batch

            # 2. Decoder input: The "Past"
            # (<sos> ... up to second to last token)
            decoder_input = batch[:-1, :]

            # 3. Targets: The "Future"
            # (First real token ... <eos>)
            targets = batch[1:, :]

            # Masks
            # Causal mask for the decoder (prevent peeking ahead)
            tgt_mask = generate_square_subsequent_mask(
                decoder_input.size(0)
            ).to(Config.DEVICE)

            # Padding masks (transpose back to [Batch, Seq] for the mask arg)
            src_padding_mask = (encoder_input == dataset.pad_idx).transpose(0, 1)
            tgt_padding_mask = (decoder_input == dataset.pad_idx).transpose(0, 1)

            optimizer.zero_grad()

            # IMPORTANT: Pass encoder_input first, decoder_input second
            logits, mu, logvar = model(
                encoder_input,
                decoder_input,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask
            )

            # Calculate Loss
            rec_loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            # KLD Annealing
            kl_weight = min(0.05, (epoch / 10) * 0.05)
            kld_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            ) / batch.size(1)

            loss = rec_loss + kl_weight * kld_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

            progress.set_postfix({
                'Loss': loss.item(),
                'KL': kld_loss.item(),
                'LR': optimizer._rate
            })

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Done. Avg Loss: {avg_loss:.4f}")
        torch.save(
            model.state_dict(),
            f"{Config.SAVE_DIR}/mug_trans_v2_ep{epoch + 1}.pth"
        )


if __name__ == "__main__":
    train()
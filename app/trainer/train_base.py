"""
Molecular Universe Generator (MUG) - Model Benchmark Suite
Author: Ali (Troxter222)
License: MIT

Comprehensive evaluation framework for comparing molecular generation models.
Tests validity, uniqueness, drug-likeness (QED), and toxicity metrics.
"""

import json
import logging
import os
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# MUG Modules
from app.config import Config
from app.core.engine import MolecularVAE
from app.core.vocab import Vocabulary

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmilesDataset(Dataset):
    def __init__(self, csv_file, vocab, max_len=200):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.max_len = max_len
        logger.info(f"Dataset loaded: {len(self.df)} molecules")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        s = str(self.df.iloc[idx]['SELFIES'])
        tokenized = self.vocab.encode(s, self.max_len)
        return torch.tensor(tokenized, dtype=torch.long)


def loss_function(recon_x, x, mu, logvar, pad_index, kl_weight):
    batch_size, seq_len, vocab_size = recon_x.shape
    recon_x = recon_x.reshape(batch_size * seq_len, vocab_size)
    x = x.reshape(batch_size * seq_len)

    ce_loss = torch.nn.CrossEntropyLoss(
        ignore_index=pad_index, reduction='sum'
    )(recon_x, x)

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return ce_loss + (kld_loss * kl_weight), ce_loss.item(), kld_loss.item()


def get_kl_weight(epoch, total_epochs):
    # Cyclical Annealing Schedule
    cycle_len = total_epochs // 2
    pos = epoch % cycle_len
    return min(1.0, (pos / float(cycle_len)) * 0.1)


def train_supervised():
    logger.info(f"Starting Supervised Training on {Config.DEVICE}")

    # 1. Load Vocabulary
    with open(Config.VOCAB_PATH, 'r') as f:
        chars = json.load(f)
    if '<sos>' not in chars:
        chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
    vocab = Vocabulary(chars)

    # 2. Data Preparation
    train_file = Config.PROCESSED_DIR / "train_selfies.csv"
    if not train_file.exists():
        logger.error(f"Training data not found at {train_file}")
        return

    dataset = SmilesDataset(train_file, vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # 3. Model Initialization
    model = MolecularVAE(
        len(vocab),
        Config.EMBED_SIZE,
        Config.HIDDEN_SIZE,
        Config.LATENT_SIZE,
        Config.NUM_LAYERS
    )
    model = model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    pad_idx = vocab.char2idx['<pad>']

    # 4. Training Loop
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        kl_weight = get_kl_weight(epoch, Config.EPOCHS)

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")

        for batch in progress:
            batch = batch.to(Config.DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            output, mu, logvar = model(inputs)
            loss, ce, kld = loss_function(
                output, targets, mu, logvar, pad_idx, kl_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({
                'Loss': loss.item() / batch.size(0),
                'KL': kld / batch.size(0)
            })

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1} Complete. Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        save_path = Config.CHECKPOINTS_DIR / f"mug_base_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Checkpoint saved: {save_path.name}")


if __name__ == "__main__":
    train_supervised()
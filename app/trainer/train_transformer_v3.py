import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
import math
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from app.core.transformer_model import MoleculeTransformer

class Config:
    FILE_DATA = 'dataset/processed_v2/transformer_train.csv'
    FILE_VOCAB = 'dataset/processed_v2/vocab_transformer.json'
    SAVE_DIR = 'checkpoints_transformer_v3'
    
    EPOCHS = 3           
    BATCH_SIZE = 24      
    ACCUM_STEPS = 8      
    LEARNING_RATE = 5e-5 # Снизил LR для стабильности
    
    SAVE_EVERY_STEPS = 5000 
    
    MAX_LEN = 100        
    D_MODEL = 256       
    NHEAD = 8           
    LAYERS = 4          
    LATENT = 128
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfiesDataset(Dataset):
    def __init__(self, csv_file, vocab_file, max_len):
        print(f"⏳ Loading CSV...")
        self.df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(self.df)} rows.")
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.max_len = max_len
        self.pad_idx = self.char2idx['<pad>']
        self.sos_idx = self.char2idx['<sos>']
        self.eos_idx = self.char2idx['<eos>']

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        s = str(self.df.iloc[idx]['selfies'])
        import selfies as sf
        try: tokens = list(sf.split_selfies(s))
        except: tokens = []
        indices = [self.sos_idx] + [self.char2idx.get(t, self.pad_idx) for t in tokens] + [self.eos_idx]
        if len(indices) > self.max_len: indices = indices[:self.max_len]
        indices += [self.pad_idx] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    print(f"🚀 STABLE TRAINING START on {Config.DEVICE}")
    
    dataset = SelfiesDataset(Config.FILE_DATA, Config.FILE_VOCAB, Config.MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    
    vocab_size = len(dataset.vocab)
    model = MoleculeTransformer(
        vocab_size, Config.D_MODEL, Config.NHEAD, 
        Config.LAYERS, Config.LAYERS, Config.D_MODEL*4, Config.LATENT
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    scaler = GradScaler() 
    
    model.train()
    
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        valid_steps = 0
        optimizer.zero_grad()
        
        progress = tqdm(dataloader, desc=f"Ep {epoch+1}/{Config.EPOCHS}")
        
        for i, batch in enumerate(progress):
            batch = batch.transpose(0, 1).to(Config.DEVICE)
            
            enc_inp = batch
            dec_inp = batch[:-1, :]
            targets = batch[1:, :]
            
            tgt_mask = generate_square_subsequent_mask(dec_inp.size(0)).to(Config.DEVICE)
            src_pad_mask = (enc_inp == dataset.pad_idx).transpose(0, 1)
            tgt_pad_mask = (dec_inp == dataset.pad_idx).transpose(0, 1)
            
            with autocast():
                logits, mu, logvar = model(
                    enc_inp, dec_inp, 
                    src_key_padding_mask=src_pad_mask,
                    tgt_key_padding_mask=tgt_pad_mask,
                    tgt_mask=tgt_mask
                )
                
                rec_loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                
                # Защита от NaN в KL
                kld_weight = min(0.005, (epoch / 5) * 0.005) # Еще меньше вес KL
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(1)
                
                loss = (rec_loss + kld_weight * kld_loss) / Config.ACCUM_STEPS

            # Проверка на NaN перед шагом назад
            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected at step {i}! Skipping batch.")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            
            if (i + 1) % Config.ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                # Очень жесткий клиппинг градиентов для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            current_loss = loss.item() * Config.ACCUM_STEPS
            total_loss += current_loss
            valid_steps += 1
            progress.set_postfix({'Loss': f"{current_loss:.4f}", 'KL': f"{kld_loss.item():.2f}"})
            
            if (i + 1) % Config.SAVE_EVERY_STEPS == 0:
                torch.save(model.state_dict(), f"{Config.SAVE_DIR}/mug_v3_ep{epoch+1}_step{i+1}.pth")
            
        avg_loss = total_loss / max(1, valid_steps)
        print(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"{Config.SAVE_DIR}/mug_v3_ep{epoch+1}.pth")

if __name__ == "__main__":
    train()
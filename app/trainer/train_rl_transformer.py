"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import json
import logging
import os
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import selfies as sf
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

# Project Imports
from app.core.transformer_model import MoleculeTransformer
from app.services.chemistry import ChemistryService

# Suppress RDKit warnings
rdBase.DisableLog('rdApp.*')

# --- LOGGING SETUP ---
LOG_FILE = 'transformer_rl.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_square_subsequent_mask(sz):
    """Generates a mask to prevent the decoder from attending to future tokens."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class RLConfig:
    """Configuration for Reinforcement Learning Fine-tuning."""
    # Paths (Using Epoch 2 as the golden standard based on previous benchmarks)
    BASE_CHECKPOINT = 'checkpoints_transformer_v2/mug_trans_v2_ep2.pth'
    SAVE_DIR = 'checkpoints_rl_transformer'
    VOCAB_FILE = 'dataset/processed/vocab_transformer.json'
    STATS_FILE = 'transformer_rl_stats.json'

    # Training Hyperparameters
    EPOCHS = 200
    BATCH_SIZE = 32        # Number of molecules per RL step
    LEARNING_RATE = 1e-5   # Low LR for fine-tuning
    PATIENCE = 30

    # Reward Function Hyperparameters
    SIGMA = 40.0           # Scalar for chemical reward
    KL_COEF = 0.2          # REINVENT-style KL penalty coefficient
    DIV_COEF = 5.0         # Diversity bonus coefficient

    # Architecture (Must match the pre-trained model)
    D_MODEL = 256
    NHEAD = 8
    LAYERS = 4
    LATENT = 128
    MAX_LEN = 120

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleVocab:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_idx = self.char2idx.get('<pad>', 0)
        self.sos_idx = self.char2idx.get('<sos>', 1)
        self.eos_idx = self.char2idx.get('<eos>', 2)

    def __len__(self):
        return len(self.vocab)

    def decode(self, indices):
        tokens = []
        for i in indices:
            idx = i.item() if torch.is_tensor(i) else i
            if idx == self.eos_idx:
                break
            if idx != self.pad_idx and idx != self.sos_idx:
                tokens.append(self.idx2char[idx])
        return "".join(tokens)


class DiversityTracker:
    """Tracks generated molecules to encourage diversity."""
    def __init__(self, max_size=2000):
        self.fingerprints = deque(maxlen=max_size)

    def get_reward(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        if not self.fingerprints:
            self.fingerprints.append(fp)
            return 1.0

        sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.fingerprints))
        max_sim = max(sims)
        self.fingerprints.append(fp)
        return 1.0 - max_sim


class RewardEngine:
    """Calculates chemical desirability."""
    @staticmethod
    def calculate(smi):
        if not smi:
            return -1.0
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return -1.0

        try:
            Chem.SanitizeMol(mol)
            props = ChemistryService.analyze_properties(mol)

            # Base QED (Quantitative Estimation of Drug-likeness)
            score = props.get('qed', 0.0)

            # Penalties & Bonuses
            if "Alerts" in str(props.get('toxicity', '')):
                score -= 0.5
            if "Yes" in str(props.get('brain', '')):  # Blood-Brain Barrier
                score += 0.3
            if props.get('sa_score', 0) > 5:
                score -= 0.2
            if props.get('mw', 0) < 200:
                score -= 0.2

            return np.clip(score, -1.0, 1.0)
        except Exception:
            return -1.0


def train_rl_transformer():
    os.makedirs(RLConfig.SAVE_DIR, exist_ok=True)
    logger.info(f"[RL] V2 SESSION STARTED on {RLConfig.DEVICE}")

    # 1. Initialize Vocabulary
    if not os.path.exists(RLConfig.VOCAB_FILE):
        logger.error(f"Vocabulary file not found: {RLConfig.VOCAB_FILE}")
        return
    vocab = SimpleVocab(RLConfig.VOCAB_FILE)

    # 2. Initialize Models
    def create_model():
        return MoleculeTransformer(
            vocab_size=len(vocab.vocab),
            d_model=RLConfig.D_MODEL,
            nhead=RLConfig.NHEAD,
            num_encoder_layers=RLConfig.LAYERS,
            num_decoder_layers=RLConfig.LAYERS,
            latent_size=RLConfig.LATENT
        ).to(RLConfig.DEVICE)

    agent = create_model()
    prior = create_model()

    # Load Pre-trained Weights
    if os.path.exists(RLConfig.BASE_CHECKPOINT):
        logger.info(f"Loading base model: {RLConfig.BASE_CHECKPOINT}")
        try:
            state = torch.load(RLConfig.BASE_CHECKPOINT, map_location=RLConfig.DEVICE)
            agent.load_state_dict(state)
            prior.load_state_dict(state)
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return
    else:
        logger.error("Base model checkpoint not found!")
        return

    # Freeze Prior (Reference Model)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(agent.parameters(), lr=RLConfig.LEARNING_RATE)
    div_tracker = DiversityTracker()

    best_avg_reward = -999.0

    # --- TRAINING LOOP ---
    for epoch in range(RLConfig.EPOCHS):
        agent.eval()
        batch_data = []
        valid_count = 0

        # Dynamic temperature adjustment
        current_temp = max(0.7, 1.2 - (epoch / RLConfig.EPOCHS) * 0.5)

        # 1. SAMPLING PHASE
        with torch.no_grad():
            for _ in range(RLConfig.BATCH_SIZE):
                try:
                    indices = agent.sample(
                        1, RLConfig.DEVICE, vocab,
                        max_len=RLConfig.MAX_LEN, temp=current_temp
                    )

                    # Convert tensor to list
                    idx_list = indices[0].cpu().tolist()
                    selfies_str = vocab.decode(torch.tensor(idx_list))
                    smi = sf.decoder(selfies_str)

                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        chem_score = RewardEngine.calculate(smi)
                        div_score = div_tracker.get_reward(mol)

                        total_reward = (chem_score * RLConfig.SIGMA) + (div_score * RLConfig.DIV_COEF)

                        # Prepare tensor for training: <sos> + sequence
                        train_indices = [vocab.sos_idx] + idx_list
                        tensor = torch.tensor(train_indices, dtype=torch.long).to(RLConfig.DEVICE)

                        batch_data.append({'tensor': tensor, 'reward': total_reward})
                        valid_count += 1
                except Exception:
                    continue

        validity = valid_count / RLConfig.BATCH_SIZE if RLConfig.BATCH_SIZE > 0 else 0
        if not batch_data:
            logger.warning(f"Epoch {epoch}: Zero valid molecules generated.")
            continue

        # 2. UPDATE PHASE (Policy Gradient)
        agent.train()
        optimizer.zero_grad()
        loss_accum = 0
        batch_rewards = []

        for item in batch_data:
            seq = item['tensor']
            reward = item['reward']

            # Prepare Inputs
            # full_seq: [Seq, 1]
            full_seq = seq.unsqueeze(1)
            # dec_input: [Seq-1, 1] (Target shifted left)
            dec_input = seq[:-1].unsqueeze(1)
            # target: [Seq-1] (Target shifted right)
            target = seq[1:].unsqueeze(1)

            # Generate Mask
            tgt_mask = generate_square_subsequent_mask(dec_input.size(0)).to(RLConfig.DEVICE)

            # --- AGENT LOG PROBS ---
            logits, _, _ = agent(full_seq, dec_input, tgt_mask=tgt_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Gather probabilities of the actual chosen tokens
            token_log_probs = log_probs.gather(2, target.unsqueeze(2)).squeeze(2)
            log_prob_agent = token_log_probs.sum()

            # --- PRIOR LOG PROBS ---
            with torch.no_grad():
                logits_p, _, _ = prior(full_seq, dec_input, tgt_mask=tgt_mask)
                log_probs_p = torch.log_softmax(logits_p, dim=-1)
                token_log_probs_p = log_probs_p.gather(2, target.unsqueeze(2)).squeeze(2)
                log_prob_prior = token_log_probs_p.sum()

            # Augmented Reward (REINVENT Algorithm)
            # R_aug = R + alpha * (log P_prior - log P_agent)
            aug_reward = reward + RLConfig.KL_COEF * (log_prob_prior - log_prob_agent).item()
            batch_rewards.append(aug_reward)

            # Loss = - R_aug * log P_agent
            # Normalized by length to prevent bias towards short sequences
            loss = -aug_reward * log_prob_agent / max(10, len(seq))
            loss_accum += loss

        avg_loss = loss_accum / len(batch_data)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # --- METRICS & SAVING ---
        avg_r_val = np.mean(batch_rewards)
        logger.info(f"Ep {epoch+1:03d} | Reward: {avg_r_val:.2f} | Validity: {validity:.0%}")

        if avg_r_val > best_avg_reward:
            best_avg_reward = avg_r_val
            torch.save(
                agent.state_dict(),
                os.path.join(RLConfig.SAVE_DIR, "mug_transformer_rl_best.pth")
            )
            logger.info(f" >> New Best Model Saved (R={avg_r_val:.2f})")

        if (epoch + 1) % 50 == 0:
            torch.save(
                agent.state_dict(),
                os.path.join(RLConfig.SAVE_DIR, f"mug_trans_rl_ep{epoch+1}.pth")
            )

    logger.info("RL Training Complete.")


if __name__ == "__main__":
    train_rl_transformer()
"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import json
import logging
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from app.config import Config
from app.core.engine import MolecularVAE
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class RLHyperparams:
    EPOCHS = 300
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    SIGMA = 50.0   # Reward scaling
    KL_COEF = 5.0  # Constraint
    DIV_COEF = 10.0  # Novelty Bonus
    # Base model path (e.g., mug_base_epoch_10.pth or best_model.pth)
    BASE_MODEL = Config.CHECKPOINTS_DIR / "mug_base_epoch_10.pth"


class DiversityTracker:
    def __init__(self, max_size=1000):
        self.fingerprints = deque(maxlen=max_size)

    def add(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        self.fingerprints.append(fp)

    def get_diversity_score(self, mol):
        if not self.fingerprints:
            return 1.0
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.fingerprints))
        return 1.0 - max(sims)


class RewardEngine:
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

            # Base QED score (0-1)
            score = props['qed']

            # Penalties and Bonuses
            # Check for specific flags returned by ChemistryService
            if "⚠️" in str(props.get('toxicity', '')):
                score -= 0.5
            if props.get('brain') == "🧠 Yes":
                score += 0.3
            if props.get('sa_score', 0) > 5:
                score -= 0.2

            return np.clip(score, -1.0, 1.0)
        except Exception:
            return -1.0


def train_reinforcement():
    logger.info(f"Starting RL Evolution on {Config.DEVICE}")

    # 1. Load Vocabulary
    with open(Config.VOCAB_PATH, 'r') as f:
        chars = json.load(f)
    if '<sos>' not in chars:
        chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
    vocab = Vocabulary(chars)

    # 2. Load Models (Agent & Prior)
    if not RLHyperparams.BASE_MODEL.exists():
        logger.error(f"Base model not found: {RLHyperparams.BASE_MODEL}")
        return

    state_dict = torch.load(RLHyperparams.BASE_MODEL, map_location=Config.DEVICE)
    vocab_size = len(vocab)

    # Initialize Agent
    agent = MolecularVAE(
        vocab_size, Config.EMBED_SIZE, Config.HIDDEN_SIZE,
        Config.LATENT_SIZE, Config.NUM_LAYERS
    ).to(Config.DEVICE)
    agent.load_state_dict(state_dict)

    # Initialize Prior (Frozen)
    prior = MolecularVAE(
        vocab_size, Config.EMBED_SIZE, Config.HIDDEN_SIZE,
        Config.LATENT_SIZE, Config.NUM_LAYERS
    ).to(Config.DEVICE)
    prior.load_state_dict(state_dict)

    for param in prior.parameters():
        param.requires_grad = False
    prior.eval()

    optimizer = optim.Adam(agent.parameters(), lr=RLHyperparams.LEARNING_RATE)
    div_tracker = DiversityTracker()

    # 3. Training Loop
    best_avg_reward = -999

    for epoch in range(RLHyperparams.EPOCHS):
        agent.eval()

        # --- GENERATION PHASE ---
        with torch.no_grad():
            indices = agent.sample(
                RLHyperparams.BATCH_SIZE, Config.DEVICE, vocab,
                max_len=150, temp=1.0
            )

        batch_data = []
        cpu_indices = indices.cpu().numpy()

        for i in range(RLHyperparams.BATCH_SIZE):
            try:
                idx_list = cpu_indices[i]
                # Trim padding
                valid_len = np.where(idx_list == vocab.char2idx['<eos>'])[0]
                end_idx = valid_len[0] + 1 if len(valid_len) > 0 else len(idx_list)
                clean_idx = idx_list[:end_idx]

                selfies_str = vocab.decode(clean_idx)
                smi = sf.decoder(selfies_str)
                mol = Chem.MolFromSmiles(smi)

                if mol:
                    chem_R = RewardEngine.calculate(smi)
                    div_R = div_tracker.get_diversity_score(mol)
                    div_tracker.add(mol)

                    total_R = (chem_R * RLHyperparams.SIGMA) + (div_R * RLHyperparams.DIV_COEF)

                    # Prepare tensor for training
                    inp = np.insert(clean_idx, 0, vocab.char2idx['<sos>'])
                    tensor = torch.tensor(inp, dtype=torch.long).to(Config.DEVICE)
                    batch_data.append({'tensor': tensor, 'reward': total_R})
            except Exception:
                continue

        if not batch_data:
            continue

        # --- UPDATE PHASE ---
        agent.train()
        optimizer.zero_grad()
        loss_accum = 0

        batch_rewards = []

        for item in batch_data:
            seq = item['tensor']
            reward = item['reward']

            inp = seq[:-1].unsqueeze(0)
            tgt = seq[1:].unsqueeze(0)

            # Agent Probabilities
            out_agent, _, _ = agent(inp)
            log_prob_agent = torch.log_softmax(out_agent, -1).gather(
                2, tgt.unsqueeze(2)
            ).squeeze(2).sum()

            # Prior Probabilities
            with torch.no_grad():
                out_prior, _, _ = prior(inp)
                log_prob_prior = torch.log_softmax(out_prior, -1).gather(
                    2, tgt.unsqueeze(2)
                ).squeeze(2).sum()

            # Augmented Reward with KL Penalty
            aug_reward = reward + RLHyperparams.KL_COEF * (
                log_prob_prior - log_prob_agent
            ).item()
            batch_rewards.append(aug_reward)

            # Policy Gradient Loss
            loss = -aug_reward * log_prob_agent / len(seq)
            loss_accum += loss

        avg_loss = loss_accum / len(batch_data)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # --- LOGGING ---
        avg_R = np.mean(batch_rewards)
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Ep {epoch + 1:03d} | Avg Reward: {avg_R:.2f} | "
                f"Batch: {len(batch_data)}"
            )

        # Save Checkpoint
        if avg_R > best_avg_reward:
            best_avg_reward = avg_R
            torch.save(
                agent.state_dict(), Config.CHECKPOINTS_DIR / "mug_rl_best.pth"
            )
            logger.info(f"New Best Model Saved (R={avg_R:.2f})")


if __name__ == "__main__":
    train_reinforcement()
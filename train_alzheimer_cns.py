"""
MUG - Alzheimer's CNS Specialist Trainer
Optimized for: Brain penetration (BBB), Low TPSA, Restrictive LogP
Target: Alzheimer (BACE1)
"""

import torch
import torch.optim as optim
import json
import os
import csv
import warnings
import numpy as np
import selfies as sf
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from concurrent.futures import ThreadPoolExecutor

from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService
from app.services.toxicity import ToxicityService

rdBase.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Config:
    BASE_MODEL_PATH = "checkpoints_specialists/alzheimer/final_alzheimer.pth" # Choose the AI which was trained 
    VOCAB_PATH = "dataset/processed/vocab_transformer.json"
    
    SAVE_DIR = "checkpoints_specialists/alzheimer_v2" # We can just add the new and new and new versions, by retraining the last AI
    LOG_FILE = f"{SAVE_DIR}/training_log.csv"
    
    STEPS = 500
    BATCH_SIZE = 16 
    LEARNING_RATE = 1e-5
    KL_COEF = 0.05
    
    TARGET_NAME = "alzheimer"
    BASELINE = 8.0 
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def patched_resolve_target_key(self, category_str):
    cat = category_str.lower()
    if "egfr" in cat: return "egfr"
    if "alzheimer" in cat or "neuro" in cat: return "alzheimer"
    if "lung" in cat or "onco" in cat: return "lung"
    if "covid" in cat or "viral" in cat: return "covid"
    return "unknown"
BiologyService._resolve_target_key = patched_resolve_target_key

class SpecialistReward:
    def __init__(self):
        self.bio = BiologyService()
        self.tox = ToxicityService()
        self.chem = ChemistryService()
        self.cache = {} 
        if not self.bio.vina_ready: raise RuntimeError("Vina not found")

    def _calculate_cns_score(self, mol):
        """
        Multiparameter Optimization for CNS drugs (BBB penetration).
        """
        score = 0.0
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        mw = Descriptors.MolWt(mol)

        # 1. TPSA (Critical for Brain) - Ideal < 70, acceptable < 90
        if tpsa < 70: score += 0.5
        elif tpsa < 90: score += 0.2
        else: score -= 0.5 # Penalty for high polarity
        
        # 2. LogP (Needs to be lipophilic but not too much)
        if 2.0 <= logp <= 4.0: score += 0.4
        elif 1.5 <= logp <= 5.0: score += 0.1
        else: score -= 0.2
        
        # 3. HBD (Hydrogen Bond Donors) - Fewer is better for BBB
        if hbd <= 1: score += 0.3
        elif hbd <= 3: score += 0.1
        
        # 4. MW (Small is better)
        if mw < 450: score += 0.2
        
        return score

    def _compute_single_molecule(self, smi):
        stats = {'valid': False, 'affinity': None, 'qed': 0.0, 'toxic': False, 'skipped': True}

        if not smi: return -1.0, stats
        mol = Chem.MolFromSmiles(smi)
        if not mol: return -1.0, stats
        stats['valid'] = True

        try: props = self.chem.analyze_properties(mol)
        except: return -1.0, stats
        stats['qed'] = props['qed']

        # MW Filter (CNS Strict)
        if props['mw'] < 150: return -0.5, stats 
        if props['mw'] > 600: return -0.5, stats 

        risks = self.tox.predict(mol)
        is_toxic = any("High" in r for r in risks)
        stats['toxic'] = is_toxic

        cns_score = self._calculate_cns_score(mol)

        if is_toxic:
            affinity = 0.0
            dock_score = 0.0
            skipped = True
        else:
            affinity = self.bio.dock_molecule(mol, Config.TARGET_NAME)

            if affinity == 0.0:
                affinity = None
                dock_score = -1.0
                skipped = True
            else:
                skipped = False

                dock_score = -(affinity + 7.0)

                if affinity > -7.0:
                    dock_score -= (affinity + 7.0) * 0.5
                else:
                    dock_score += 0.5

        
        stats['affinity'] = affinity
        stats['skipped'] = skipped
        tox_penalty = 2.0 if is_toxic else 0.0
        
        # CNS Score is heavily weighted here
        final_score = dock_score + cns_score + (props['qed'] * 0.5) - tox_penalty
        return final_score, stats

    def get_reward_batch(self, smiles_list):
        scores = [0.0] * len(smiles_list)
        stats_list = [{} for _ in smiles_list]
        indices = []
        for i, smi in enumerate(smiles_list):
            if smi in self.cache: scores[i], stats_list[i] = self.cache[smi]
            else: indices.append(i)
        
        if indices:
            to_compute = [smiles_list[i] for i in indices]
            with ThreadPoolExecutor(max_workers=8) as ex:
                results = list(ex.map(self._compute_single_molecule, to_compute))
            for idx_batch, (sc, st) in zip(indices, results):
                scores[idx_batch] = sc
                stats_list[idx_batch] = st
                if st.get('valid'): self.cache[smiles_list[idx_batch]] = (sc, st)
        return torch.tensor(scores, device=Config.DEVICE, dtype=torch.float32), stats_list

def train():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    with open(Config.LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'smiles', 'reward', 'affinity', 'qed', 'toxic'])

    with open(Config.VOCAB_PATH) as f: vocab = Vocabulary(json.load(f))
    
    agent = MoleculeTransformer(29, 256, 8, 4, 4, 1024, 128).to(Config.DEVICE)
    prior = MoleculeTransformer(29, 256, 8, 4, 4, 1024, 128).to(Config.DEVICE)
    
    state = torch.load(Config.BASE_MODEL_PATH, map_location=Config.DEVICE)
    new_state = {k.replace('fc_z', 'fc_latent_to_hidden'): v for k, v in state.items()}
    agent.load_state_dict(new_state, strict=False)
    prior.load_state_dict(new_state, strict=False)
    prior.eval()
    
    optimizer = optim.Adam(agent.parameters(), lr=Config.LEARNING_RATE)
    scorer = SpecialistReward()
    
    print(f"🧠 STARTING ALZHEIMER CNS SPECIALIST | Target: {Config.TARGET_NAME} | Baseline: {Config.BASELINE}")

    for step in range(Config.STEPS):
        agent.train()
        with torch.no_grad():
            seqs = agent.sample(Config.BATCH_SIZE, Config.DEVICE, vocab, max_len=120, temperature=1.0)
        
        smiles = [sf.decoder(vocab.decode(s)) or "" for s in seqs.cpu().numpy()]
        rewards, stats = scorer.get_reward_batch(smiles)
        
        if len(rewards) > 1:
            rewards_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_norm = rewards

        inp = seqs.t()
        logits_ag, _, _ = agent(inp, inp[:-1, :])
        with torch.no_grad(): logits_pr, _, _ = prior(inp, inp[:-1, :])
            
        tgt = inp[1:, :]
        log_p_ag = logits_ag.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        log_p_pr = logits_pr.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        
        loss = -(rewards_norm * log_p_ag).mean() + Config.KL_COEF * (log_p_ag - log_p_pr).pow(2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        valid_affs = [s['affinity'] for s in stats if isinstance(s.get('affinity'), (int, float)) and s['affinity'] < 0]
        avg_aff = sum(valid_affs)/len(valid_affs) if valid_affs else 0
        
        print(f"Step {step+1:03d} | Rw: {rewards.mean():.2f} | Aff: {avg_aff:.2f}", end="\r")
        
        best_idx = np.argmax(rewards.cpu().numpy())
        with open(Config.LOG_FILE, 'a', newline='') as f:
            st = stats[best_idx]
            csv.writer(f).writerow([step+1, smiles[best_idx], f"{rewards[best_idx]:.3f}", st.get('affinity',0), f"{st.get('qed',0):.2f}", st.get('toxic')])
            
        if (step+1) % 50 == 0:
            torch.save(agent.state_dict(), f"{Config.SAVE_DIR}/model_step{step+1}.pth")

    torch.save(agent.state_dict(), f"{Config.SAVE_DIR}/final_alzheimer.pth")
    print("\nDone.")

if __name__ == "__main__":
    train()
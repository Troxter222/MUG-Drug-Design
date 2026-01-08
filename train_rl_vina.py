"""
Molecular Universe Generator (MUG) - Vina RL Benchmark
Author: Ali (Troxter222)
License: MIT
"""

import torch
import torch.optim as optim
import json
import os
import csv
import numpy as np
import selfies as sf
from rdkit import Chem, rdBase
from concurrent.futures import ThreadPoolExecutor

from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService
from app.services.toxicity import ToxicityService
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", module="meeko")

rdBase.DisableLog('rdApp.*')

# --- MONKEY PATCH FOR BIOLOGY SERVICE ---
# Fixes target resolution logic to ensure 'covid' and 'egfr' are mapped correctly
def patched_resolve_target_key(self, category_str):
    cat = category_str.lower()
    if "egfr" in cat: 
        return "egfr"
    if "alzheimer" in cat or "neuro" in cat: 
        return "alzheimer"
    if "lung" in cat or "onco" in cat: 
        return "lung"
    if "covid" in cat or "viral" in cat: 
        return "covid"
    return "unknown"

BiologyService._resolve_target_key = patched_resolve_target_key

# --- CONFIGURATION ---
class VinaRLConfig:
    BASE_MODEL_PATH = "checkpoints_rl_finetuned/mug_rl_neuro_v2.pth"
    VOCAB_PATH = "dataset/processed/vocab_transformer.json"
    
    SAVE_DIR = "checkpoints_specialists/covid/model_v1"
    LOG_FILE = "checkpoints_specialists/covid/model_v1/training_log.csv"
    
    # Training Hyperparameters
    STEPS = 500
    BATCH_SIZE = 16 
    LEARNING_RATE = 1e-5
    KL_COEF = 0.05
    
    # Weights
    WEIGHT_DOCK = 1.0
    WEIGHT_QED = 0.5
    WEIGHT_TOX = 2.0
    
    # Target Configuration
    # Key: Target name, Value: Baseline affinity (positive float)
    # Only using 'covid' as it is the only one confirmed working in previous logs
    TARGET_SPECS = {
        "covid": 6.5,  # MPro target
        # "egfr": 8.5, # Uncomment if .pdbqt file exists
        # "lung": 7.0, # Uncomment if .pdbqt file exists
    }
    
    TARGET_ORDER = list(TARGET_SPECS.keys())
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- REWARD CALCULATOR ---
class VinaReward:
    def __init__(self):
        self.bio = BiologyService()
        self.tox = ToxicityService()
        self.chem = ChemistryService()
        self.cache = {} 
        
        if not self.bio.vina_ready:
            raise RuntimeError("Vina executable not found.")

    def _compute_single_molecule(self, args):
        """
        Calculates reward components for a single molecule.
        Args: (smiles, target_name, baseline)
        """
        smi, target_name, baseline = args
        
        # Initialize default stats to prevent KeyError
        stats = {
            'valid': False,
            'affinity': 0.0,
            'qed': 0.0,
            'toxic': False,
            'mw_fail': False,
            'skipped': True,
            'target': target_name
        }

        # 1. Validity Check
        if not smi:
            return -1.0, stats
        
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return -1.0, stats
        
        stats['valid'] = True

        # 2. Properties Analysis
        try:
            props = self.chem.analyze_properties(mol)
        except Exception:
             stats['valid'] = False
             return -1.0, stats
        
        stats['qed'] = props['qed']

        # Molecular Weight Filter (Softened)
        # Penalty if MW < 150 (fragments) or MW > 750 (too large)
        mw = props['mw']
        mw_penalty = 0.0
        
        if mw < 150:
             mw_penalty = 0.2
             stats['mw_fail'] = True
        elif mw > 750:
             stats['mw_fail'] = True
             return -0.5, stats # Hard stop for huge molecules

        # 3. Toxicity Check
        risks = self.tox.predict(mol)
        is_toxic = any("High" in r for r in risks)
        stats['toxic'] = is_toxic

        # 4. Docking Simulation
        if is_toxic:
            affinity = 0.0
            dock_score = 0.0 
            skipped_dock = True
        else:
            affinity = self.bio.dock_molecule(mol, target_name)
            
            # Handle case where Vina fails (returns 0.0) without punishing model excessively
            if affinity == 0.0:
                dock_score = 0.0
                skipped_dock = True
            else:
                # Reward formula: -(Affinity + Baseline)
                dock_score = -(affinity + baseline)
                skipped_dock = False
        
        stats['affinity'] = affinity
        stats['skipped'] = skipped_dock

        # 5. Final Score Aggregation
        tox_penalty = 2.0 if is_toxic else 0.0
        
        final_score = (dock_score * VinaRLConfig.WEIGHT_DOCK) + \
                      (props['qed'] * VinaRLConfig.WEIGHT_QED) - \
                      tox_penalty - mw_penalty
                      
        return final_score, stats

    def get_reward_batch(self, smiles_list, target_name):
        baseline = VinaRLConfig.TARGET_SPECS[target_name]
        
        scores = [0.0] * len(smiles_list)
        stats_list = [{} for _ in smiles_list]
        
        indices_to_compute = []
        args_to_compute = []
        hits = 0
        
        # Check cache
        for i, smi in enumerate(smiles_list):
            cache_key = (smi, target_name)
            if cache_key in self.cache:
                sc, st = self.cache[cache_key]
                scores[i] = sc
                stats_list[i] = st
                hits += 1
            else:
                indices_to_compute.append(i)
                args_to_compute.append((smi, target_name, baseline))
        
        # Compute missing
        if args_to_compute:
            with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
                results = list(executor.map(self._compute_single_molecule, args_to_compute))
            
            for idx_in_batch, (score, stats) in zip(indices_to_compute, results):
                smi = smiles_list[idx_in_batch]
                scores[idx_in_batch] = score
                stats_list[idx_in_batch] = stats
                
                # Cache valid molecules
                if smi and stats.get('valid') and not stats.get('mw_fail'):
                    self.cache[(smi, target_name)] = (score, stats)
                    
        return torch.tensor(scores, device=VinaRLConfig.DEVICE, dtype=torch.float32), stats_list, hits

# --- HELPERS ---
def load_adapter(model, path):
    print(f"Loading adapter: {path}")
    state = torch.load(path, map_location=VinaRLConfig.DEVICE)
    new_state = {}
    for k, v in state.items():
        if 'fc_z' in k:
            new_state[k.replace('fc_z', 'fc_latent_to_hidden')] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)

def init_csv_logging():
    with open(VinaRLConfig.LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'target', 'best_smiles', 'reward', 'affinity', 'qed', 'toxic'])

def log_step_to_csv(step, target, smiles_list, rewards, stats_list):
    best_idx = np.argmax(rewards.cpu().numpy())
    if best_idx >= len(stats_list): 
        return
    best_stat = stats_list[best_idx]
    
    with open(VinaRLConfig.LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            step, 
            target,
            smiles_list[best_idx], 
            f"{rewards[best_idx].item():.4f}", 
            best_stat.get('affinity', 0), 
            f"{best_stat.get('qed', 0):.2f}", 
            best_stat.get('toxic', False)
        ])

# --- MAIN TRAINING LOOP ---
def train():
    os.makedirs(VinaRLConfig.SAVE_DIR, exist_ok=True)
    init_csv_logging()
    
    with open(VinaRLConfig.VOCAB_PATH) as f: 
        vocab = Vocabulary(json.load(f))
    
    vocab_size = 29
    agent = MoleculeTransformer(vocab_size, 256, 8, 4, 4, 1024, 128).to(VinaRLConfig.DEVICE)
    prior = MoleculeTransformer(vocab_size, 256, 8, 4, 4, 1024, 128).to(VinaRLConfig.DEVICE)
    
    try:
        load_adapter(agent, VinaRLConfig.BASE_MODEL_PATH)
        load_adapter(prior, VinaRLConfig.BASE_MODEL_PATH)
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    prior.eval()
    optimizer = optim.Adam(agent.parameters(), lr=VinaRLConfig.LEARNING_RATE)
    scorer = VinaReward()
    
    print(f"Starting Vina RL. Targets: {VinaRLConfig.TARGET_ORDER}")

    for step in range(VinaRLConfig.STEPS):
        # Cyclical target selection
        target_idx = step % len(VinaRLConfig.TARGET_ORDER)
        current_target = VinaRLConfig.TARGET_ORDER[target_idx]
        
        agent.train()
        
        # 1. Sample
        with torch.no_grad():
            seqs = agent.sample(VinaRLConfig.BATCH_SIZE, VinaRLConfig.DEVICE, vocab, max_len=120, temperature=1.0)
        
        smiles_batch = []
        for s in seqs.cpu().numpy():
            try: 
                smiles_batch.append(sf.decoder(vocab.decode(s)) or "")
            except Exception: 
                smiles_batch.append("")
            
        # 2. Calculate Rewards
        rewards_raw, stats, hits = scorer.get_reward_batch(smiles_batch, current_target)
        
        # Normalize rewards
        if len(rewards_raw) > 1:
            r_mean = rewards_raw.mean()
            r_std = rewards_raw.std()
            rewards_normalized = (rewards_raw - r_mean) / (r_std + 1e-8)
        else:
            rewards_normalized = rewards_raw

        # 3. Compute Loss
        inp = seqs.t()
        logits_ag, _, _ = agent(inp, inp[:-1, :])
        with torch.no_grad():
            logits_pr, _, _ = prior(inp, inp[:-1, :])
            
        tgt = inp[1:, :]
        log_p_ag = logits_ag.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        log_p_pr = logits_pr.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        
        # Policy Gradient + MSE Regularization
        pg_loss = - (rewards_normalized * log_p_ag).mean()
        mse_reg = (log_p_ag - log_p_pr).pow(2).mean()
        
        loss = pg_loss + (VinaRLConfig.KL_COEF * mse_reg)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        # 4. Logging
        # Safe extraction of affinity values
        valid_affs = [
            s.get('affinity', 0.0) 
            for s in stats 
            if s.get('valid') and not s.get('skipped') and s.get('affinity', 0.0) < 0
        ]
        avg_aff = sum(valid_affs)/len(valid_affs) if valid_affs else 0.0
        
        print(f"Step {step+1:03d} | [{current_target}] | Rw: {rewards_raw.mean().item():.2f} | Aff: {avg_aff:.1f} | Hits: {hits}", end="\r")
        
        log_step_to_csv(step+1, current_target, smiles_batch, rewards_raw, stats)
        
        if (step+1) % 50 == 0:
            torch.save(agent.state_dict(), f"{VinaRLConfig.SAVE_DIR}/mug_vina_step{step+1}.pth")

    torch.save(agent.state_dict(), f"{VinaRLConfig.SAVE_DIR}/mug_vina_final.pth")
    print("\nTraining Complete.")

if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import selfies as sf
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, AllChem

# Imports
from app.config import Config
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.toxicity import ToxicityService

rdBase.DisableLog('rdApp.*')

# --- CONFIGURATION ---
class UniversalConfig:
    # 1. Начинаем с твоей лучшей текущей модели (Neuro V2)
    # Когда обучишь 14М, просто поменяешь этот путь на новую модель!
    BASE_MODEL_PATH = "checkpoints_rl_finetuned/mug_rl_neuro_v2.pth"
    
    # Путь к словарю (тот же, что и у модели)
    VOCAB_PATH = "dataset/processed/vocab_transformer.json"
    
    SAVE_DIR = "checkpoints_rl_universal"
    
    # Параметры RL
    STEPS = 600          # Чуть дольше, чтобы переучить на крупные молекулы
    BATCH_SIZE = 64      
    LEARNING_RATE = 1e-5 
    SIGMA = 50           # Сила награды
    KL_COEF = 0.15       # Удерживаем синтаксис
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HELPERS ---
def load_checkpoint_safe(model, path):
    print(f"🔄 Loading weights: {path}")
    state_dict = torch.load(path, map_location=UniversalConfig.DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'fc_z' in k: new_state_dict[k.replace('fc_z', 'fc_latent_to_hidden')] = v
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model

# --- PROXY BINDING CALCULATOR ---
def calculate_proxy_affinity(mol):
    """
    Быстрая оценка энергии связывания без запуска Vina (для скорости обучения).
    Основана на физике: Вес + Водородные связи + Роторы.
    """
    mw = Descriptors.MolWt(mol)
    hb = Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)
    rotors = Lipinski.NumRotatableBonds(mol)
    aromatic = len(mol.GetSubstructMatches(Chem.MolFromSmarts("a")))
    
    # Базовая формула: -0.025 * MW (чем тяжелее, тем лучше, до предела)
    # Бонус за ароматику (pi-stacking)
    score = - (mw * 0.025) - (aromatic * 0.1)
    
    # Штраф за гибкость (энтропия)
    score += (rotors * 0.15)
    
    # Ограничиваем "мечты"
    return max(-13.0, min(-4.0, score))

# --- ADVANCED REWARD FUNCTION ---
class UniversalReward:
    def __init__(self):
        self.tox = ToxicityService()
    
    def get_reward(self, smiles_list):
        rewards = []
        for smi in smiles_list:
            score = 0.0
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol or mol.GetNumAtoms() < 12: 
                    rewards.append(-2.0) # Усиливаем штраф за мелочь
                    continue
                
                props = ChemistryService.analyze_properties(mol)
                mw = props['mw']
                
                # --- 1. ПРЯМАЯ ТЯГА К ВЕСУ (Target: 400 Da) ---
                # Даем бонус, который растет вместе с весом до 450
                if mw < 350:
                    score += (mw / 70.0) # Молекула 280 Da даст +4.0
                elif 350 <= mw <= 500:
                    score += 8.0 # Огромный джекпот за идеальный вес
                
                # --- 2. СИЛА СВЯЗИ (Affinity) ---
                proxy_affinity = calculate_proxy_affinity(mol)
                # Экспоненциальная награда: за каждый шаг к -8.0 даем больше
                if proxy_affinity < -8.5:
                    score += 15.0 # Фантастический результат
                elif proxy_affinity < -7.5:
                    score += 8.0
                elif proxy_affinity < -6.5:
                    score += 4.0
                elif proxy_affinity < -5.5:
                    score += 1.0

                # --- 3. ФАРМАКОФОРНЫЕ ГРУППЫ ---
                # Лекарства любят фтор (F) и азотные циклы
                if any(atom.GetSymbol() == 'F' for atom in mol.GetAtoms()):
                    score += 1.5
                
                # --- 4. ШТРАФ ЗА ПЛОХУЮ ХИМИЮ ---
                if props['qed'] < 0.3:
                    score -= 3.0 # Молекула не должна быть "грязной"
                
                risks = self.tox.predict(mol)
                if any("High" in r for r in risks): 
                    score -= 5.0 

            except:
                score = -1.0
            
            rewards.append(score)
        return torch.tensor(rewards, device=UniversalConfig.DEVICE)
    
# --- TRAINER ---
def train_universal():
    # 1. Init
    try:
        with open(UniversalConfig.VOCAB_PATH, 'r') as f: 
            chars = json.load(f)
        vocab = Vocabulary(chars)
    except Exception:
        # Fallback to create dummy vocab to init model (size will be fixed by checkpoint)
        vocab = Vocabulary(['<pad>']*100) 

    # 2. Get Real Vocab Size from Checkpoint
    ckpt = torch.load(UniversalConfig.BASE_MODEL_PATH, map_location=UniversalConfig.DEVICE)
    real_size = ckpt['embedding.weight'].shape[0]
    print(f"🧠 Model Vocab Size: {real_size}")
    
    # 3. Models
    agent = MoleculeTransformer(real_size, 256, 8, 4, 4, 1024, 128).to(UniversalConfig.DEVICE)
    prior = MoleculeTransformer(real_size, 256, 8, 4, 4, 1024, 128).to(UniversalConfig.DEVICE)
    
    load_checkpoint_safe(agent, UniversalConfig.BASE_MODEL_PATH)
    load_checkpoint_safe(prior, UniversalConfig.BASE_MODEL_PATH)
    prior.eval()
    
    optimizer = optim.Adam(agent.parameters(), lr=UniversalConfig.LEARNING_RATE)
    scorer = UniversalReward()
    
    print("\n🚀 Starting UNIVERSAL RL (Affinity + Safety)...")
    
    # 4. Loop
    for step in range(UniversalConfig.STEPS):
        agent.train()
        
        # Sample
        with torch.no_grad():
            seqs = agent.sample(UniversalConfig.BATCH_SIZE, UniversalConfig.DEVICE, vocab, max_len=120, temperature=1.3)
        
        # Decode & Reward
        smiles_batch = []
        for s in seqs.cpu().numpy():
            try: 
                smiles_batch.append(sf.decoder(vocab.decode(s)) or "")
            except Exception: 
                smiles_batch.append("")
            
        rewards = scorer.get_reward(smiles_batch)
        
        # Loss
        inp = seqs.t()
        enc, dec = inp, inp[:-1, :]
        tgt = inp[1:, :]
        
        logits_ag, _, _ = agent(enc, dec)
        log_p_ag = logits_ag.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        
        with torch.no_grad():
            logits_pr, _, _ = prior(enc, dec)
            log_p_pr = logits_pr.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
            
        loss = - ((rewards + UniversalConfig.KL_COEF * (log_p_pr - log_p_ag)) * log_p_ag).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        if step % 20 == 0:
            avg_r = rewards.mean().item()
            print(f"Step {step} | Avg Reward: {avg_r:.4f}")
            valid = [s for s in smiles_batch if len(s) > 5]
            if valid: 
                mol = Chem.MolFromSmiles(valid[0])
                aff = calculate_proxy_affinity(mol) if mol else 0
                print(f"   Sample: {valid[0]}")
                print(f"   Proxy Affinity: {aff:.2f} kcal/mol (Target < -8.0)")

    # Save
    os.makedirs(UniversalConfig.SAVE_DIR, exist_ok=True)
    save_path = f"{UniversalConfig.SAVE_DIR}/mug_universal_v1.pth"
    torch.save(agent.state_dict(), save_path)
    print(f"\n✅ UNIVERSAL Model Saved: {save_path}")

if __name__ == "__main__":
    train_universal()
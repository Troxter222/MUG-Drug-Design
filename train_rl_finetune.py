import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import selfies as sf
from rdkit import Chem, rdBase

# Импорты твоего проекта
from app.config import Config
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.toxicity import ToxicityService

# Глушим RDKit
rdBase.DisableLog('rdApp.*')

# --- CONFIG ---
class RLConfig:
    # 1. Путь к модели V2 (проверенной)
    BASE_MODEL_PATH = "checkpoints_transformer_v2/mug_trans_v2_ep11.pth"
    
    # 2. ВАЖНО: Путь к СТАРОМУ словарю (без _v2), на котором училась V2
    # Если файла нет, скрипт попытается восстановить размер из чекпоинта
    VOCAB_PATH = "dataset/processed/vocab_transformer.json"
    
    SAVE_DIR = "checkpoints_rl_finetuned"
    
    STEPS = 500          
    BATCH_SIZE = 64      
    LEARNING_RATE = 1e-5 
    SIGMA = 50           
    KL_COEF = 0.2        
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- HELPER: Адаптивная загрузка весов ---
def load_checkpoint_safe(model, path):
    """Грузит старые чекпоинты в новый код, исправляя имена слоев."""
    print(f"🔄 Loading adapter for: {path}")
    state_dict = torch.load(path, map_location=RLConfig.DEVICE)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        # Ремаппинг старых имен
        if 'fc_z' in k:
            new_key = k.replace('fc_z', 'fc_latent_to_hidden')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    # Грузим с strict=False, чтобы игнорировать мелочи
    model.load_state_dict(new_state_dict, strict=False)
    return model

# --- REWARD FUNCTION ---
class RewardFunction:
    def __init__(self):
        self.tox = ToxicityService()
    
    def get_reward(self, smiles_list):
        rewards = []
        for smi in smiles_list:
            score = 0.0
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    rewards.append(0.0) 
                    continue
                
                # 1. QED
                props = ChemistryService.analyze_properties(mol)
                score += props['qed']
                
                # 2. CNS Rules (Neuro Reward)
                if "High" in props['cns_prob']:
                    score += 1.0
                elif "Low" in props['cns_prob']:
                    score -= 0.5
                
                # 3. Toxicity Penalty
                risks = self.tox.predict(mol)
                if risks: score -= 0.5
                
            except:
                score = 0.0
            
            rewards.append(score)
        return torch.tensor(rewards, device=RLConfig.DEVICE)

# --- VALIDITY TEST ---
def test_validity():
    print("🔍 Testing Model & Vocab compatibility...")
    
    # 1. Загружаем словарь
    try:
        with open(RLConfig.VOCAB_PATH, 'r') as f:
            chars = json.load(f)
        vocab = Vocabulary(chars)
        print(f"📚 Vocab loaded: {len(vocab)} tokens")
    except FileNotFoundError:
        print(f"⚠️ Vocab file not found at {RLConfig.VOCAB_PATH}. Using fallback size from checkpoint.")
        # Заглушка, чтобы просто инициировать модель (размер возьмем из чекпоинта)
        vocab = Vocabulary(['<pad>', '<sos>', '<eos>'] + ['C']*100) 

    # 2. Загружаем веса, чтобы узнать реальный размер словаря
    state_dict = torch.load(RLConfig.BASE_MODEL_PATH, map_location=RLConfig.DEVICE)
    real_vocab_size = state_dict['embedding.weight'].shape[0]
    print(f"⚖️ Checkpoint expects vocab size: {real_vocab_size}")
    
    # Если размер файла словаря не совпадает с моделью, это проблема
    if len(vocab) != real_vocab_size:
        print(f"⚠️ WARNING: Vocab file size ({len(vocab)}) != Model size ({real_vocab_size})")
        print("➡️ Will rely on model weights dimensions.")
        # Хак: создаем словарь нужного размера (может работать некорректно при декодировании, если json не тот)
        # Но если json просто потерялся, это позволит хотя бы запустить
    
    # 3. Создаем модель
    agent = MoleculeTransformer(real_vocab_size, 256, 8, 4, 4, 1024, 128).to(RLConfig.DEVICE)
    load_checkpoint_safe(agent, RLConfig.BASE_MODEL_PATH)
    agent.eval()
    
    # 4. Тест генерации
    print("🧪 Generating test sample...")
    indices = agent.sample(5, RLConfig.DEVICE, vocab, max_len=100)
    valid = 0
    for idx in indices.cpu().numpy():
        try:
            # Декодируем
            smi = sf.decoder(vocab.decode(idx))
            if smi and Chem.MolFromSmiles(smi):
                valid += 1
                print(f"   ✅ {smi}")
            else:
                print("   ❌ Invalid syntax")
        except Exception:
            print("   ❌ Decode error (wrong vocab file?)")
    
    if valid == 0:
        print("🛑 STOP! Model produces garbage. Check VOCAB_PATH matches the checkpoint training data.")
        return None, None
        
    print(f"✅ Validity Check: {valid}/5. System Green.")
    return agent, vocab

# --- RL TRAINER ---
def train_rl():
    agent, vocab = test_validity()
    if not agent: return

    # Prior (Frozen copy)
    prior = MoleculeTransformer(len(vocab), 256, 8, 4, 4, 1024, 128).to(RLConfig.DEVICE)
    load_checkpoint_safe(prior, RLConfig.BASE_MODEL_PATH)
    prior.eval()
    
    optimizer = optim.Adam(agent.parameters(), lr=RLConfig.LEARNING_RATE)
    scorer = RewardFunction()
    
    print("\n🚀 Starting Reinforcement Learning...")
    
    for step in range(RLConfig.STEPS):
        agent.train()
        
        # A. Sampling
        with torch.no_grad():
            seqs = agent.sample(RLConfig.BATCH_SIZE, RLConfig.DEVICE, vocab, max_len=100, temperature=1.0)
        
        # B. Scoring
        smiles_batch = []
        for s in seqs.cpu().numpy():
            try:
                txt = vocab.decode(s)
                smi = sf.decoder(txt)
                smiles_batch.append(smi if smi else "")
            except: smiles_batch.append("")
            
        rewards = scorer.get_reward(smiles_batch)
        
        # C. Loss
        inp = seqs.t()
        enc_inp, dec_inp = inp, inp[:-1, :]
        tgt = inp[1:, :]
        
        logits_agent, _, _ = agent(enc_inp, dec_inp)
        log_p_agent = logits_agent.log_softmax(dim=-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(dim=0)
        
        with torch.no_grad():
            logits_prior, _, _ = prior(enc_inp, dec_inp)
            log_p_prior = logits_prior.log_softmax(dim=-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(dim=0)
        
        augmented_reward = rewards + RLConfig.KL_COEF * (log_p_prior - log_p_agent)
        loss = - (augmented_reward * log_p_agent).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Reward: {rewards.mean().item():.4f} | Loss: {loss.item():.4f}")
            valid_mols = [s for s in smiles_batch if len(s)>2]
            if valid_mols: print(f"   Example: {valid_mols[0]}")

    # Save
    torch.save(agent.state_dict(), f"{RLConfig.SAVE_DIR}/mug_rl_neuro_v2.pth")
    print("✅ RL Training Complete.")

if __name__ == "__main__":
    os.makedirs(RLConfig.SAVE_DIR, exist_ok=True)
    train_rl()
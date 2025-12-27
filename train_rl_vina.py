import torch
import torch.optim as optim
import numpy as np
import json
import os
import selfies as sf
from rdkit import Chem, rdBase
from concurrent.futures import ThreadPoolExecutor # Для параллельного докинга

# Imports
from app.config import Config
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService # Теперь используем реальный класс
from app.services.toxicity import ToxicityService

rdBase.DisableLog('rdApp.*')

# --- CONFIG FOR VINA TRAINING ---
class VinaRLConfig:
    # База: берем нашу лучшую Neuro V2 или Universal
    BASE_MODEL_PATH = "checkpoints_rl_finetuned/mug_rl_neuro_v2.pth"
    VOCAB_PATH = "dataset/processed/vocab_transformer.json"
    
    # Сохраняем в отдельную папку
    SAVE_DIR = "checkpoints_vina_egfr"
    
    # Меньше шагов, так как каждый шаг очень "умный"
    STEPS = 200          
    BATCH_SIZE = 32      # Уменьшил батч, чтобы CPU не захлебнулся
    LEARNING_RATE = 5e-6 # Очень аккуратно меняем веса
    
    # ВАЖНО: Цель для докинга (должна быть в targets_config.json)
    TARGET_CATEGORY = "lung" # EGFR
    TARGET_KEY = "lung"      # Ключ для BiologyService

    KL_COEF = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- PHYSICS-BASED REWARD ---
class VinaReward:
    def __init__(self):
        self.bio = BiologyService()
        self.tox = ToxicityService()
        self.chem = ChemistryService()
        
        # Проверяем, готова ли Vina
        if not self.bio.vina_ready:
            raise RuntimeError("❌ Vina.exe not found! Cannot train using physics.")
            
    def calculate_single_score(self, smi):
        """Считает награду для одной молекулы (будет запущено в потоке)."""
        if not smi: return 0.0
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if not mol: return 0.0
            
            # 1. Свойства
            props = self.chem.analyze_properties(mol)
            
            # Если молекула совсем "кривая" (слишком мелкая/огромная),
            # даже не тратим время CPU на докинг.
            if props['mw'] < 200 or props['mw'] > 650:
                return -0.5
            
            # 2. REAL DOCKING (Самое важное)
            # Это вернет реальные ккал/моль (например, -8.5)
            affinity = self.bio.dock_molecule(mol, VinaRLConfig.TARGET_CATEGORY)
            
            # 3. Превращаем энергию в награду (Score)
            # Нам нужно -9.0 и ниже.
            # Если -6.0 -> Score 0
            # Если -9.0 -> Score 3.0
            # Если -11.0 -> Score 5.0
            docking_reward = max(0, -(affinity + 6.0)) 
            
            # 4. Штрафы
            risks = self.tox.predict(mol)
            tox_penalty = 1.0 if any("High" in r for r in risks) else 0.0
            
            # Итоговая формула
            final_score = docking_reward - tox_penalty + props['qed']
            
            return final_score
            
        except Exception:
            return 0.0

    def get_reward_batch(self, smiles_list):
        """Параллельный запуск Vina для всего батча"""
        print(f"   🧬 Docking {len(smiles_list)} molecules in parallel...", end="\r")
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            rewards = list(executor.map(self.calculate_single_score, smiles_list))
            
        return torch.tensor(rewards, device=VinaRLConfig.DEVICE)

# --- HELPERS ---
def load_adapter(model, path):
    print(f"🔄 Adapter load: {path}")
    state = torch.load(path, map_location=VinaRLConfig.DEVICE)
    new_state = {}
    for k, v in state.items():
        if 'fc_z' in k: new_state[k.replace('fc_z', 'fc_latent_to_hidden')] = v
        else: new_state[k] = v
    model.load_state_dict(new_state, strict=False)

# --- TRAINING LOOP ---
def train_vina():
    os.makedirs(VinaRLConfig.SAVE_DIR, exist_ok=True)
    
    # Init
    with open(VinaRLConfig.VOCAB_PATH) as f: vocab = Vocabulary(json.load(f))
    
    # Models
    vocab_size = 29 # Или len(vocab), проверь по предыдущему уроку!
    agent = MoleculeTransformer(vocab_size, 256, 8, 4, 4, 1024, 128).to(VinaRLConfig.DEVICE)
    prior = MoleculeTransformer(vocab_size, 256, 8, 4, 4, 1024, 128).to(VinaRLConfig.DEVICE)
    
    # Загружаем веса (у тебя может быть 29 или 118 токенов, проверь vocab_size!)
    # Если падает на mismatch, ставь vocab_size=29 вручную как в прошлом фиксе
    try:
        load_adapter(agent, VinaRLConfig.BASE_MODEL_PATH)
        load_adapter(prior, VinaRLConfig.BASE_MODEL_PATH)
    except:
        print("⚠️ Vocab mismatch suspected. Assuming old vocab size 29.")
        agent = MoleculeTransformer(29, 256, 8, 4, 4, 1024, 128).to(VinaRLConfig.DEVICE)
        prior = MoleculeTransformer(29, 256, 8, 4, 4, 1024, 128).to(VinaRLConfig.DEVICE)
        load_adapter(agent, VinaRLConfig.BASE_MODEL_PATH)
        load_adapter(prior, VinaRLConfig.BASE_MODEL_PATH)

    prior.eval()
    optimizer = optim.Adam(agent.parameters(), lr=VinaRLConfig.LEARNING_RATE)
    scorer = VinaReward()
    
    print(f"\n🚀 Starting PHYSICS-BASED RL for target: {VinaRLConfig.TARGET_CATEGORY}")
    print("⚠️ This will take time. Watch the reward grow!")

    for step in range(VinaRLConfig.STEPS):
        agent.train()
        
        # 1. Sample
        with torch.no_grad():
            seqs = agent.sample(VinaRLConfig.BATCH_SIZE, VinaRLConfig.DEVICE, vocab, max_len=120, temperature=1.0)
        
        # 2. Decode
        smiles_batch = []
        for s in seqs.cpu().numpy():
            try: smiles_batch.append(sf.decoder(vocab.decode(s)) or "")
            except: smiles_batch.append("")
            
        # 3. REAL VINA SCORING (Slow part)
        rewards = scorer.get_reward_batch(smiles_batch)
        
        # 4. PPO Loss
        inp = seqs.t()
        logits_ag, _, _ = agent(inp, inp[:-1, :])
        
        with torch.no_grad():
            logits_pr, _, _ = prior(inp, inp[:-1, :])
            
        # Выбираем вероятности сгенерированных токенов
        tgt = inp[1:, :]
        log_p_ag = logits_ag.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        log_p_pr = logits_pr.log_softmax(-1).gather(2, tgt.unsqueeze(2)).squeeze(2).sum(0)
        
        loss = - ((rewards + VinaRLConfig.KL_COEF * (log_p_pr - log_p_ag)) * log_p_ag).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        avg_r = rewards.mean().item()
        print(f"Step {step+1}/{VinaRLConfig.STEPS} | Avg Reward: {avg_r:.4f} | Loss: {loss.item():.2f}")
        
        # Save every 50 steps
        if (step+1) % 50 == 0:
            path = f"{VinaRLConfig.SAVE_DIR}/mug_egfr_hunter_step{step+1}.pth"
            torch.save(agent.state_dict(), path)
            print(f"💾 Checkpoint: {path}")

    # Final Save
    final_path = f"{VinaRLConfig.SAVE_DIR}/mug_egfr_hunter_final.pth"
    torch.save(agent.state_dict(), final_path)
    print(f"✅ Training Complete! Model saved to {final_path}")

if __name__ == "__main__":
    train_vina()
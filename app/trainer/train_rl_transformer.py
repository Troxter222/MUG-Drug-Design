import torch
import torch.optim as optim
import selfies as sf
import json
import os
import numpy as np
import logging
from collections import deque
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem

# Глушим RDKit
rdBase.DisableLog('rdApp.*')

# Импорты проекта
from app.core.transformer_model import MoleculeTransformer
from app.services.chemistry import ChemistryService

# Логирование
LOG_FILE = 'transformer_rl.log'
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
stream_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# --- МАСКА ДЛЯ ДЕКОДЕРА ---
def generate_square_subsequent_mask(sz):
    """Маска, чтобы декодер не видел будущее"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class RLConfig:
    # ПУТИ (Берем Эпоху 2 - Золотой стандарт)
    BASE_CHECKPOINT = 'checkpoints_transformer_v2/mug_trans_v2_ep2.pth'
    SAVE_DIR = 'checkpoints_rl_transformer'
    VOCAB_FILE = 'dataset/processed/vocab_transformer.json'
    STATS_FILE = 'transformer_rl_stats.json'
    
    # ТРЕНИРОВКА
    EPOCHS = 200          
    BATCH_SIZE = 32        # Количество молекул в "эпохе" RL
    LEARNING_RATE = 1e-5   # Аккуратный файн-тюнинг
    PATIENCE = 30          
    
    # НАГРАДА
    SIGMA = 40.0           # Чуть уменьшил агрессию (было 50)
    KL_COEF = 0.2          # REINVENT-style KL penalty
    DIV_COEF = 5.0
    
    # АРХИТЕКТУРА (Должна совпадать с train_transformer_v2.py !!!)
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
        
    def __len__(self): return len(self.vocab)
    
    def decode(self, indices):
        tokens = []
        for i in indices:
            idx = i.item() if torch.is_tensor(i) else i
            if idx == self.eos_idx: break
            if idx != self.pad_idx and idx != self.sos_idx:
                tokens.append(self.idx2char[idx])
        return "".join(tokens)

class DiversityTracker:
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
    @staticmethod
    def calculate(smi):
        if not smi: return -1.0
        mol = Chem.MolFromSmiles(smi)
        if not mol: return -1.0
        
        try:
            Chem.SanitizeMol(mol)
            props = ChemistryService.analyze_properties(mol)
            
            # Базовый QED
            score = props['qed']
            
            # Penalties & Bonuses
            if "Alerts" in str(props['toxicity']): score -= 0.5
            if "Yes" in str(props['brain']): score += 0.3
            if props['sa_score'] > 5: score -= 0.2
            if props['mw'] < 200: score -= 0.2 # Штраф за мелочь
            
            return np.clip(score, -1.0, 1.0)
        except:
            return -1.0

def train_rl_transformer():
    os.makedirs(RLConfig.SAVE_DIR, exist_ok=True)
    logger.info(f"[RL] V2 SESSION STARTED on {RLConfig.DEVICE}")
    
    # 1. Init Vocab
    vocab = SimpleVocab(RLConfig.VOCAB_FILE)
    
    # 2. Init Models
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
    
    if os.path.exists(RLConfig.BASE_CHECKPOINT):
        logger.info(f"[INFO] Loading base: {RLConfig.BASE_CHECKPOINT}")
        try:
            state = torch.load(RLConfig.BASE_CHECKPOINT, map_location=RLConfig.DEVICE)
            agent.load_state_dict(state)
            prior.load_state_dict(state)
        except Exception as e:
            logger.error(f"[FATAL] Size mismatch or load error: {e}")
            logger.error("Check RLConfig architecture params vs Pretraining!")
            return
    else:
        logger.error("[ERROR] Base model not found!")
        return

    # Freeze Prior
    prior.eval()
    for p in prior.parameters(): p.requires_grad = False
    
    optimizer = optim.Adam(agent.parameters(), lr=RLConfig.LEARNING_RATE)
    div_tracker = DiversityTracker()
    
    best_avg_reward = -999
    
    # --- TRAINING LOOP ---
    for epoch in range(RLConfig.EPOCHS):
        agent.eval()
        batch_data = []
        valid_count = 0

        current_temp = max(0.7, 1.2 - (epoch / RLConfig.EPOCHS) * 0.5)
        
        # 1. GENERATION (Sampling)
        with torch.no_grad():
            for _ in range(RLConfig.BATCH_SIZE):
                try:
                    indices = agent.sample(1, RLConfig.DEVICE, vocab, max_len=RLConfig.MAX_LEN, temp=current_temp)
                    
                    # Convert [1, Seq] -> List
                    idx_list = indices[0].cpu().tolist()
                    selfies_str = vocab.decode(torch.tensor(idx_list))
                    smi = sf.decoder(selfies_str)
                    
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        chem_R = RewardEngine.calculate(smi)
                        div_R = div_tracker.get_reward(mol)
                        
                        total_R = (chem_R * RLConfig.SIGMA) + (div_R * RLConfig.DIV_COEF)
                        
                        # Prepare tensor: <sos> ... <eos>
                        # indices already has body. Add SOS.
                        train_indices = [vocab.sos_idx] + idx_list
                        tensor = torch.tensor(train_indices, dtype=torch.long).to(RLConfig.DEVICE)
                        
                        batch_data.append({'tensor': tensor, 'reward': total_R})
                        valid_count += 1
                except: continue
        
        validity = valid_count / RLConfig.BATCH_SIZE
        if not batch_data:
            logger.warning(f"Ep {epoch}: Zero valid molecules.")
            continue
        
        # 2. UPDATE (Policy Gradient)
        agent.train()
        optimizer.zero_grad()
        loss_accum = 0
        batch_rewards = []
        
        # Generating Mask once (assuming max length fits) - optimization
        
        for item in batch_data:
            seq = item['tensor']
            reward = item['reward']
            
            # Data setup
            # full_seq: [Seq, 1] (Encoder sees ALL)
            full_seq = seq.unsqueeze(1) 
            
            # dec_input: [Seq-1, 1] (Decoder sees Past)
            dec_input = seq[:-1].unsqueeze(1)
            
            # target: [Seq-1] (Next tokens)
            target = seq[1:].unsqueeze(1)
            
            # FIX #3: Custom Mask
            tgt_mask = generate_square_subsequent_mask(dec_input.size(0)).to(RLConfig.DEVICE)
            
            # --- AGENT PROB ---
            # FIX #2: Correct forward pass logic
            logits, _, _ = agent(full_seq, dec_input, tgt_mask=tgt_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, target.unsqueeze(2)).squeeze(2)
            log_prob_agent = token_log_probs.sum()
            
            # --- PRIOR PROB ---
            with torch.no_grad():
                logits_p, _, _ = prior(full_seq, dec_input, tgt_mask=tgt_mask)
                log_probs_p = torch.log_softmax(logits_p, dim=-1)
                token_log_probs_p = log_probs_p.gather(2, target.unsqueeze(2)).squeeze(2)
                log_prob_prior = token_log_probs_p.sum()
            
            # Augmented Reward (REINVENT)
            # R_aug = R + alpha * (log P_prior - log P_agent)
            aug_reward = reward + RLConfig.KL_COEF * (log_prob_prior - log_prob_agent).item()
            batch_rewards.append(aug_reward)
            
            # Loss = - R_aug * log P_agent
            # Normalization by length to avoid bias towards short molecules
            loss = -aug_reward * log_prob_agent / max(10, len(seq)) 
            loss_accum += loss
            
        avg_loss = loss_accum / len(batch_data)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        
        # --- LOGS ---
        avg_R = np.mean(batch_rewards)
        logger.info(f"Ep {epoch+1:03d} | Reward: {avg_R:.2f} | Valid: {validity:.0%}")
        
        if avg_R > best_avg_reward:
            best_avg_reward = avg_R
            torch.save(agent.state_dict(), os.path.join(RLConfig.SAVE_DIR, "mug_transformer_rl_best.pth"))
            logger.info(f" [SAVE] New Best Model (R={avg_R:.2f})")
            
        if (epoch+1) % 50 == 0:
             torch.save(agent.state_dict(), os.path.join(RLConfig.SAVE_DIR, f"mug_trans_rl_ep{epoch+1}.pth"))

    logger.info(" [DONE] RL Training Complete.")

if __name__ == "__main__":
    train_rl_transformer()
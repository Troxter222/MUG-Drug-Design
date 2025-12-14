import torch
import pandas as pd
import numpy as np
import json
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors

# Импортируем обе архитектуры
from app.core.engine import MolecularVAE as GRU_Model
from app.core.transformer_model import MoleculeTransformer as Trans_Model
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService

# --- НАСТРОЙКИ БИТВЫ ---
SAMPLES_COUNT = 500  # Сколько молекул генерируем
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. КОНФИГ GRU (Твой старый чемпион)
GRU_CFG = {
    "name": "Deep GRU (RL)",
    "path": "data/checkpoints_rl/mug_rl_best.pth", # Проверь путь!
    "vocab": "data/processed/vocab_selfies.json",
    "params": {"vocab_size": 0, "embed_size": 64, "hidden_size": 256, "latent_size": 128, "num_layers": 3},
    "type": "gru"
}

# 2. КОНФИГ TRANSFORMER (Твой новый чемпион)
TRANS_CFG = {
    "name": "Transformer V2 (RL)",
    "path": "checkpoints_rl_transformer/mug_transformer_rl_best.pth", # Проверь путь!
    "vocab": "dataset/processed/vocab_transformer.json",
    "params": {
        "vocab_size": 0, 
        "d_model": 256,   # <--- Как в обучении RL
        "nhead": 8, 
        "num_encoder_layers": 4, 
        "num_decoder_layers": 4, 
        "latent_size": 128, # <--- Как в обучении RL
        "dim_feedforward": 1024 
    },
    "type": "trans"
}

def load_vocab(path):
    with open(path, 'r') as f: chars = json.load(f)
    if '<sos>' not in chars: chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
    return Vocabulary(chars)

def evaluate_model(config):
    print(f"\n🤖 Тестирование: {config['name']}...")
    
    if not os.path.exists(config['path']):
        print(f"❌ Файл {config['path']} не найден!")
        return None

    # 1. Load Vocab
    vocab = load_vocab(config['vocab'])
    config['params']['vocab_size'] = len(vocab)
    
    # 2. Load Model
    if config['type'] == 'gru':
        model = GRU_Model(**config['params']).to(DEVICE)
    else:
        model = Trans_Model(**config['params']).to(DEVICE)
        
    try:
        model.load_state_dict(torch.load(config['path'], map_location=DEVICE))
    except Exception as e:
        print(f"❌ Ошибка загрузки весов: {e}")
        return None
        
    model.eval()
    
    # 3. Generation
    valid_mols = []
    unique_smiles = set()
    novelty_count = 0 # (Упрощенно, без сверки с train)
    
    qeds = []
    sas = []
    mws = []
    tox_free = 0
    
    print(f"⚗️ Генерация {SAMPLES_COUNT} молекул...")
    with torch.no_grad():
        for _ in tqdm(range(SAMPLES_COUNT)):
            try:
                if config['type'] == 'gru':
                    # GRU sample (предполагаем, что он возвращает тензор [1, len])
                    idx = model.sample(1, DEVICE, vocab, max_len=120, temp=0.8)
                    if isinstance(idx, torch.Tensor): idx = idx.cpu().numpy()[0]
                else:
                    # Transformer sample
                    idx = model.sample(1, DEVICE, vocab, max_len=120, temp=0.8)
                    if isinstance(idx, torch.Tensor): idx = idx[0].cpu().numpy()
                
                # Decode
                selfies = vocab.decode(idx)
                smi = sf.decoder(selfies)
                if not smi: continue
                
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    valid_mols.append(mol)
                    unique_smiles.add(smi)
                    
                    # Metrics
                    props = ChemistryService.analyze_properties(mol)
                    qeds.append(props['qed'])
                    mws.append(props['mw'])
                    sas.append(props['sa_score'])
                    
                    if "✅" in props['toxicity']:
                        tox_free += 1
            except: continue
            
    # 4. Aggregate
    total = SAMPLES_COUNT
    valid = len(valid_mols)
    
    metrics = {
        "Model": config['name'],
        "Validity": (valid / total) * 100,
        "Uniqueness": (len(unique_smiles) / valid * 100) if valid > 0 else 0,
        "Avg QED": np.mean(qeds) if qeds else 0,
        "Avg SA": np.mean(sas) if sas else 0,
        "Tox Free": (tox_free / valid * 100) if valid > 0 else 0,
        "Avg MW": np.mean(mws) if mws else 0
    }
    
    return metrics

import os

if __name__ == "__main__":
    results = []
    
    # Test GRU
    res_gru = evaluate_model(GRU_CFG)
    if res_gru: results.append(res_gru)
    
    # Test Transformer
    res_trans = evaluate_model(TRANS_CFG)
    if res_trans: results.append(res_trans)
    
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("🏆 FINAL BENCHMARK RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        df.to_csv("final_benchmark.csv", index=False)
        print("\n💾 Сохранено в final_benchmark.csv")
    else:
        print("❌ Тесты не прошли.")
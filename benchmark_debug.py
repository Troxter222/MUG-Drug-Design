import torch
import glob
import json
import os
import sys
import pandas as pd
import numpy as np
import selfies as sf
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors, AllChem

# --- ФИКС ИМПОРТОВ ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты проекта
from app.core.engine import MolecularVAE
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary

# --- НАСТРОЙКИ ---
SEARCH_DIRS = [
    "checkpoints_transformer", 
    "checkpoints_rl_transformer"
]
# Проверяем файлы словарей
VOCAB_TRANS = "dataset/processed/vocab_transformer.json"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BenchmarkDebugger:
    def __init__(self):
        print(f"🔧 Debug Mode. Device: {DEVICE}")

    def load_model(self, path):
        print(f"\n📂 Загрузка: {path}")
        
        # 1. Загрузка словаря
        try:
            with open(VOCAB_TRANS, 'r') as f: chars = json.load(f)
            # Фикс спецсимволов
            if '<sos>' not in chars: chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
            vocab = Vocabulary(chars)
            print(f"   📚 Словарь: {len(vocab)} токенов")
        except Exception as e:
            print(f"   ❌ Ошибка словаря: {e}")
            return None, None

        # 2. Загрузка модели
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            
            # Проверка размеров
            saved_vocab = checkpoint['embedding.weight'].shape[0]
            if saved_vocab != len(vocab):
                print(f"   ⚠️ MISMATCH: Vocab={len(vocab)}, Model={saved_vocab}. Using Model size.")
                current_vocab_len = saved_vocab
            else:
                current_vocab_len = len(vocab)

            # Инициализация (Трансформер)
            model = MoleculeTransformer(
                vocab_size=current_vocab_len, 
                d_model=128, 
                nhead=4, 
                num_encoder_layers=3, 
                num_decoder_layers=3, 
                latent_size=64
            )
                
            model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()
            print("   ✅ Веса загружены успешно.")
            return model, vocab
            
        except Exception as e:
            print(f"   ❌ Ошибка загрузки весов: {e}")
            return None, None

    def test_generation(self, model, vocab):
        print("   🧪 Попытка генерации...")
        try:
            with torch.no_grad():
                # Пробуем сгенерировать 1 молекулу и смотрим, где упадет
                indices = model.sample(DEVICE, vocab, max_len=100)
                
                # Проверяем, что вернулось
                if isinstance(indices, torch.Tensor):
                    indices = indices.cpu().numpy().tolist()
                
                print(f"   🔢 Indices received: {indices[:5]}...")
                
                decoded = vocab.decode(torch.tensor(indices))
                print(f"   🔤 Decoded SELFIES: {decoded}")
                
                smi = sf.decoder(decoded)
                print(f"   🧬 SMILES: {smi}")
                
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    print("   🎉 SUCCESS: Valid Molecule!")
                    return True
                else:
                    print("   ⚠️ Invalid Molecule (RDKit failed)")
                    return False
                    
        except Exception as e:
            print(f"   🔥 CRASH DURING GENERATION: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(self):
        files = []
        for d in SEARCH_DIRS:
            files.extend(glob.glob(os.path.join(d, "*.pth")))
            
        if not files:
            print("❌ Файлы моделей не найдены!")
            return

        # Тестируем только первый найденный файл для дебага
        print(f"🔎 Найдено {len(files)} файлов. Тестируем первый...")
        f = files[0] # Берем первый попавшийся
        
        model, vocab = self.load_model(f)
        if model:
            self.test_generation(model, vocab)

if __name__ == "__main__":
    BenchmarkDebugger().run()
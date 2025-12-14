import torch
import json
import os
import glob
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import matplotlib.pyplot as plt
from app.core.transformer_model import MoleculeTransformer

# --- КОНФИГУРАЦИЯ (Должна совпадать с train_transformer.py) ---
class TestConfig:
    VOCAB_FILE = 'dataset/processed/vocab_transformer.json'
    CHECKPOINT_DIR = 'checkpoints_transformer'
    
    # Архитектура (Обязана совпадать с обучением!)
    D_MODEL = 128
    NHEAD = 4
    LAYERS = 3
    LATENT = 64
    MAX_LEN = 150
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleVocab:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_idx = self.char2idx.get('<pad>', 0)
        self.sos_idx = self.char2idx.get('<sos>', 1)
        self.eos_idx = self.char2idx.get('<eos>', 2)

    def decode(self, indices):
        tokens = []
        for i in indices:
            idx = i.item() if torch.is_tensor(i) else i
            if idx == self.eos_idx: 
                break
            if idx != self.pad_idx and idx != self.sos_idx:
                tokens.append(self.idx2char[idx])
        return "".join(tokens)

def get_latest_checkpoint():
    """Находит самый свежий файл весов"""
    files = glob.glob(f"{TestConfig.CHECKPOINT_DIR}/*.pth")
    if not files:
        return None
    # Сортируем по времени создания (свежие в конце)
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def test():
    print("🔬 ЗАПУСК ТЕСТА ТРАНСФОРМЕРА...")
    
    # 1. Загрузка словаря
    if not os.path.exists(TestConfig.VOCAB_FILE):
        print(f"❌ Словарь не найден: {TestConfig.VOCAB_FILE}")
        return
    vocab = SimpleVocab(TestConfig.VOCAB_FILE)
    
    # 2. Поиск модели
    ckpt_path = get_latest_checkpoint()
    if not ckpt_path:
        print(f"❌ Чекпоинты не найдены в {TestConfig.CHECKPOINT_DIR}. Подожди окончания 1-й эпохи.")
        return
    
    print(f"📂 Загружаю веса: {ckpt_path}")
    
    # 3. Инициализация
    model = MoleculeTransformer(
        vocab_size=len(vocab.vocab),
        d_model=TestConfig.D_MODEL,
        nhead=TestConfig.NHEAD,
        num_encoder_layers=TestConfig.LAYERS,
        num_decoder_layers=TestConfig.LAYERS,
        latent_size=TestConfig.LATENT
    ).to(TestConfig.DEVICE)
    
    model.load_state_dict(torch.load(ckpt_path, map_location=TestConfig.DEVICE))
    model.eval()
    
    # 4. Генерация
    NUM_SAMPLES = 50
    print(f"⚗️ Генерирую {NUM_SAMPLES} молекул...")
    
    valid_mols = []
    valid_smiles = []
    
    with torch.no_grad():
        for _ in range(NUM_SAMPLES):
            # Генерируем 1 штуку (можно батчами, но так проще дебажить)
            indices = model.sample(TestConfig.DEVICE, vocab, max_len=TestConfig.MAX_LEN)
            selfies_str = vocab.decode(indices)
            
            try:
                # Декодируем SELFIES -> SMILES
                smi = sf.decoder(selfies_str)
                if not smi: 
                    continue
                
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    valid_mols.append(mol)
                    valid_smiles.append(smi)
            except Exception:
                continue

    # 5. Результаты
    validity = (len(valid_mols) / NUM_SAMPLES) * 100
    unique = len(set(valid_smiles))
    unique_ratio = (unique / len(valid_smiles) * 100) if valid_smiles else 0
    
    print("\n📊 ОТЧЕТ О ТЕСТИРОВАНИИ:")
    print(f"✅ Валидность: {validity:.1f}% (Цель > 80%)")
    print(f"🦄 Уникальность: {unique_ratio:.1f}%")
    
    if valid_mols:
        print("\n🧪 Примеры генерации:")
        for i in range(min(5, len(valid_smiles))):
            print(f"{i+1}. {valid_smiles[i]}")
            
        # Рисуем сетку
        img = Draw.MolsToGridImage(valid_mols[:9], molsPerRow=3, subImgSize=(300, 300), legends=[f"Mol {i+1}" for i in range(len(valid_mols[:9]))])
        img.save("transformer_test_results.png")
        print("\n🖼 Картинка сохранена в transformer_test_results.png")
        
        # Считаем средний QED
        qeds = [Descriptors.qed(m) for m in valid_mols]
        avg_qed = sum(qeds) / len(qeds)
        print(f"💊 Средний QED (Drug-likeness): {avg_qed:.2f}")
    else:
        print("⚠️ Модель сгенерировала только мусор. Нужно учить дольше.")

if __name__ == "__main__":
    test()
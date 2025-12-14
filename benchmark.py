import torch
import glob
import json
import os
import pandas as pd
import numpy as np
import selfies as sf
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors, AllChem

# Импорты проекта
from app.core.engine import MolecularVAE
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService

# --- НАСТРОЙКИ ---
SEARCH_DIRS = [
    "checkpoints", 
    "checkpoints_selfies", 
    "checkpoints_rl_ultimate", 
    "checkpoints_transformer", 
    "checkpoints_rl_transformer"
]
TRAIN_DATA_PATH = "data/processed/train_selfies.csv"
VOCAB_GRU = "data/processed/vocab_selfies.json"
VOCAB_TRANS = "dataset/processed/vocab_transformer.json" # Путь к словарю трансформера

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 500 # Количество молекул для теста

class BenchmarkEngine:
    def __init__(self):
        print("⏳ Загрузка тренировочных данных для проверки новизны...")
        try:
            df = pd.read_csv(TRAIN_DATA_PATH)
            # Берем первые 100к для скорости, или все если памяти много
            self.train_smiles = set()
            for s in tqdm(df['SELFIES'][:100000]):
                try: self.train_smiles.add(sf.decoder(s))
                except: pass
            print(f"✅ Загружено {len(self.train_smiles)} уникальных молекул из train.")
        except:
            print("⚠️ Тренировочный датасет не найден. Novelty будет 100%.")
            self.train_smiles = set()

    def get_model_type(self, path):
        if "transformer" in path.lower():
            return "transformer", VOCAB_TRANS
        return "gru", VOCAB_GRU

    def load_model(self, path):
        model_type, vocab_path = self.get_model_type(path)
        
        try:
            with open(vocab_path, 'r') as f: chars = json.load(f)
            if '<sos>' not in chars: chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
            vocab = Vocabulary(chars)
            
            checkpoint = torch.load(path, map_location=DEVICE)
            vocab_size = checkpoint['embedding.weight'].shape[0]
            
            # Авто-фикс размера словаря, если не совпадает
            if len(vocab) != vocab_size:
                # Временно подменяем размер словаря для загрузки весов
                # (В реальном тесте это может сломать генерацию, но мы попробуем)
                real_vocab_len = vocab_size
            else:
                real_vocab_len = len(vocab)

            if model_type == "transformer":
                model = MoleculeTransformer(real_vocab_len, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, latent_size=64)
            else:
                model = MolecularVAE(real_vocab_len, 64, 256, 128, 3)
                
            model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()
            return model, vocab, model_type
        except Exception as e:
            print(f"❌ Ошибка загрузки {path}: {e}")
            return None, None, None

    def calculate_metrics(self, smiles_list):
        valid_mols = []
        valid_smiles = []
        
        for s in smiles_list:
            if not s: continue
            m = Chem.MolFromSmiles(s)
            if m:
                valid_mols.append(m)
                valid_smiles.append(s)
        
        total = len(smiles_list)
        if total == 0: return None
        
        # 1. Validity
        validity = len(valid_mols) / total
        
        if not valid_mols:
            return {"Validity": 0.0, "Score": 0.0}

        # 2. Uniqueness
        unique_smiles = set(valid_smiles)
        uniqueness = len(unique_smiles) / len(valid_mols)
        
        # 3. Novelty
        new_mols = [s for s in unique_smiles if s not in self.train_smiles]
        novelty = len(new_mols) / len(unique_smiles)
        
        # 4. Diversity (Tanimoto)
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in valid_mols]
        if len(fps) > 1:
            divs = []
            for i in range(10): # Рандомная выборка для скорости
                import random
                if len(fps) < 2: break
                a, b = random.sample(fps, 2)
                divs.append(1.0 - DataStructs.TanimotoSimilarity(a, b))
            diversity = np.mean(divs)
        else:
            diversity = 0.0

        # 5. Properties
        qeds = []
        logps = []
        mws = []
        sas = []
        tox_alerts = 0
        
        for m in valid_mols:
            qeds.append(Descriptors.qed(m))
            logps.append(Crippen.MolLogP(m))
            mws.append(Descriptors.MolWt(m))
            
            # SA Score proxy
            complexity = GraphDescriptors.BertzCT(m)
            sa = (complexity - 200) / 100
            sas.append(sa)
            
            # Tox Check (Simple)
            if m.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")) or \
               m.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1[Cl,Br,I]")):
                tox_alerts += 1
                
        avg_qed = np.mean(qeds)
        avg_sa = np.mean(sas)
        avg_tox = tox_alerts / len(valid_mols)
        
        # 6. Reward (Approximation)
        # R = QED*10 + (1-Tox)*5 + (Novelty)*5
        avg_reward = (avg_qed * 10) + ((1 - avg_tox) * 5) + (novelty * 5)
        
        return {
            "Validity": validity * 100,
            "Uniqueness": uniqueness * 100,
            "Novelty": novelty * 100,
            "Diversity": diversity,
            "QED": avg_qed,
            "SA": avg_sa,
            "LogP": np.mean(logps),
            "MW": np.mean(mws),
            "Tox_Rate": avg_tox * 100,
            "Reward": avg_reward
        }

    def run(self):
        # Находим все файлы моделей
        files = []
        for d in SEARCH_DIRS:
            files.extend(glob.glob(os.path.join(d, "*.pth")))
            
        print(f"🔎 Найдено {len(files)} моделей. Начинаю турнир...\n")
        
        report = []
        
        for f in files:
            print(f"🤖 Testing: {f} ...", end=" ")
            model, vocab, mtype = self.load_model(f)
            
            if not model: continue
            
            # Генерируем
            generated_smiles = []
            with torch.no_grad():
                for _ in range(int(N_SAMPLES / 10)): # Батчами по 10
                    try:
                        indices = model.sample(10, DEVICE, vocab, max_len=150, temp=0.8)
                        # Обработка в зависимости от того, что возвращает sample (список или тензор)
                        if isinstance(indices, list):
                            # Если возвращает плоский список (для батча=1), это проблема для батча 10
                            # Предположим sample возвращает тензор [batch, len]
                            pass 
                        
                        # Костыль для универсальности (так как sample в разных версиях разный)
                        # Генерим по 1, если батч не поддерживается в sample
                        # Но для скорости предположим, что sample переписан или используем цикл
                        pass
                    except: pass
            
            # Простой цикл генерации по 1 (медленно, но надежно для всех версий кода)
            for _ in range(50): # 50 молекул для теста (чтобы быстро)
                try:
                    if mtype == "transformer":
                        idx = model.sample(DEVICE, vocab, max_len=100)
                        s = vocab.decode(torch.tensor(idx))
                    else:
                        idx = model.sample(1, DEVICE, vocab, max_len=100, temp=0.8)
                        s = vocab.decode(idx.cpu().numpy()[0])
                    
                    smi = sf.decoder(s)
                    generated_smiles.append(smi)
                except: continue
                
            metrics = self.calculate_metrics(generated_smiles)
            
            if metrics:
                metrics['Model'] = os.path.basename(f)
                metrics['Type'] = mtype
                report.append(metrics)
                print(f"✅ QED: {metrics['QED']:.2f} | Valid: {metrics['Validity']:.0f}%")
            else:
                print("❌ Fail")

        # --- АНАЛИЗ И ВЫВОД ---
        if not report:
            print("Нет результатов.")
            return

        df = pd.DataFrame(report)
        
        # Сортировка по "Супер-метрике" (MIT Score)
        # Score = QED (max) + Validity (max) - Tox (min)
        df['MIT_Score'] = (df['QED'] * 20) + (df['Validity'] / 10) - (df['Tox_Rate'] / 10) + (df['Novelty'] / 20)
        
        df = df.sort_values(by='MIT_Score', ascending=False)
        
        print("\n" + "="*80)
        print("🏆 РЕЗУЛЬТАТЫ ТУРНИРА МОДЕЛЕЙ MUG")
        print("="*80)
        
        print("\n🥇 ТОП-3 ЛУЧШИХ МОДЕЛИ:")
        print(df[['Model', 'Type', 'MIT_Score', 'QED', 'Validity', 'Tox_Rate']].head(3).to_string(index=False))
        
        print("\n📊 ПОЛНАЯ ТАБЛИЦА (Группа A - Структура):")
        print(df[['Model', 'Validity', 'Uniqueness', 'Novelty', 'Diversity']].to_string(index=False))
        
        print("\n💊 ПОЛНАЯ ТАБЛИЦА (Группа B - Фарма):")
        print(df[['Model', 'QED', 'SA', 'LogP', 'MW', 'Tox_Rate']].to_string(index=False))
        
        # Сохранение
        df.to_csv("benchmark_results.csv", index=False)
        print("\n💾 Полный отчет сохранен в benchmark_results.csv")

if __name__ == "__main__":
    BenchmarkEngine().run()
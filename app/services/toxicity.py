import os
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

class ToxicityService:
    # Путь к моделям
    MODEL_DIR = Path("data/models/tox21")
    
    # Красивые названия для отчета
    LABELS = {
        'SR-ATAD5': 'Genotoxicity (DNA Damage)',
        'NR-AhR': 'Toxin Response (AhR)',
        'SR-HSE': 'Cellular Stress (Heat Shock)',
        'SR-MMP': 'Mitochondrial Toxicity',
        'NR-AR': 'Androgen Disruption',
        'NR-ER': 'Estrogen Disruption',
        'SR-p53': 'Cancer Risk (p53)'
    }

    def __init__(self):
        self.models = {}
        self.loaded = False
        self._load_models()

    def _load_models(self):
        if not self.MODEL_DIR.exists():
            print("⚠️ Toxicity models not found. Run train_tox_ai.py first.")
            return
            
        print("☢️ Loading AI-Toxicology Models...")
        try:
            for task_file in self.MODEL_DIR.glob("*.pkl"):
                task_name = task_file.stem
                self.models[task_name] = joblib.load(task_file)
            self.loaded = True
            print(f"✅ Loaded {len(self.models)} toxicity classifiers.")
        except Exception as e:
            print(f"❌ Failed to load tox models: {e}")

    def predict(self, mol):
        """
        Возвращает список рисков с вероятностями.
        """
        if not self.loaded or not mol:
            return []

        # Векторизация (Fingerprint)
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            fp = gen.GetFingerprint(mol)
        except Exception:
            # Старый способ (если версия RDKit старая)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_arr = np.array(fp).reshape(1, -1)
        
        risks = []
        
        # Прогон по всем моделям
        for task, model in self.models.items():
            # Получаем вероятность класса "1" (Токсичен)
            prob = model.predict_proba(fp_arr)[0][1]
            
            # Если вероятность > 50% (или выше для строгости), считаем риском
            if prob > 0.5:
                # Берем красивое имя или код
                name = self.LABELS.get(task, task)
                
                # Уровень опасности
                severity = "High" if prob > 0.8 else "Medium" if prob > 0.65 else "Low"
                icon = "🔴" if prob > 0.8 else "🟠" if prob > 0.65 else "🟡"
                
                risks.append(f"{icon} {name}: {prob*100:.0f}% ({severity})")
                
        return risks
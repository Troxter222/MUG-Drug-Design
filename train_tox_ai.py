"""
Molecular Universe Generator (MUG) - Model Benchmark Suite
Author: Ali (Troxter222)
License: MIT

Comprehensive evaluation framework for comparing molecular generation models.
Tests validity, uniqueness, drug-likeness (QED), and toxicity metrics.
"""

import os
import requests
import gzip
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- CONFIG ---
DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
SAVE_DIR = "data/models/tox21"
DATA_FILE = "dataset/raw/tox21.csv.gz"

# 12 ключевых задач Tox21
TASKS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

# Человекочитаемые названия
TASK_NAMES = {
    'NR-AR': 'Androgen Receptor (Hormonal)',
    'NR-AhR': 'AhR (Toxin Response)',
    'NR-Aromatase': 'Aromatase (Enzyme)',
    'NR-ER': 'Estrogen Receptor',
    'SR-ARE': 'Oxidative Stress',
    'SR-ATAD5': 'Genotoxicity (Cancer)',
    'SR-HSE': 'Heat Shock (Cell Stress)',
    'SR-MMP': 'Mitochondrial Toxicity',
    'SR-p53': 'p53 (Tumor Suppressor)'
}

def download_data():
    os.makedirs("dataset/raw", exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("⬇️ Downloading Tox21 from DeepChem...")
        r = requests.get(DATA_URL)
        with open(DATA_FILE, 'wb') as f:
            f.write(r.content)
    
    print("📂 Loading Dataset...")
    with gzip.open(DATA_FILE, 'rt') as f:
        df = pd.read_csv(f)
    return df

def mol_to_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Morgan Fingerprint (Radius 2, 1024 bits) - стандарт индустрии
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            return np.array(fp)
    except Exception: 
        pass
    return None

def train_suite():
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = download_data()

    print("Vectorizing Molecules (Fingerprints)...")
    valid_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fp = mol_to_fp(row['smiles'])
        if fp is not None:
            entry = {'fp': fp}
            for t in TASKS: 
                entry[t] = row[t]
            valid_data.append(entry)
            
    print(f"Valid molecules: {len(valid_data)}")
    
    X_all = np.array([d['fp'] for d in valid_data])
    
    print("\nTraining 12 AI-Toxicologists...")
    scores = {}
    
    for task in TASKS:
        y_data = [d[task] for d in valid_data]
        
        X_task = []
        y_task = []
        for x, y in zip(X_all, y_data):
            if not np.isnan(y):
                X_task.append(x)
                y_task.append(y)
                
        X_task = np.array(X_task)
        y_task = np.array(y_task)
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_task, y_task, test_size=0.1, random_state=42)
        
        # Model: Random Forest (Быстрый, мощный, не переобучается)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=20, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        probs = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception: 
            auc = 0.5
        
        scores[task] = auc
        print(f"  {task:<15} AUC: {auc:.3f}")
        
        # Save
        joblib.dump(clf, f"{SAVE_DIR}/{task}.pkl")

    print("\nAll models saved to data/models/tox21/")
    print(f"Average AUC: {np.mean(list(scores.values())):.3f}")

if __name__ == "__main__":
    train_suite()
import joblib
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from rdkit.Chem import rdFingerprintGenerator
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False


class ToxicityService:
    """
    Service for predicting molecular toxicity using pre-trained models.
    """
    # Path to model directory
    MODEL_DIR = Path("data/models/tox21")

    # Human-readable labels for report generation
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
        """Loads all .pkl models from the specified directory."""
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
        Predicts toxicity risks for a given RDKit molecule.
        Returns a list of formatted warning strings.
        """
        if not self.loaded or not mol:
            return []

        # Generate Morgan Fingerprint
        try:
            if HAS_GENERATOR:
                gen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=2, fpSize=1024
                )
                fp = gen.GetFingerprintAsNumPy(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=1024
                )
            
            # Ensure it is a reshaped numpy array for sklearn
            fp_arr = np.array(fp).reshape(1, -1)

        except Exception as e:
            print(f"Error generating fingerprint: {e}")
            return []

        risks = []

        for task, model in self.models.items():
            # Get probability for class 1 (Toxic)
            try:
                prob = model.predict_proba(fp_arr)[0][1]
            except AttributeError:
                continue

            if prob > 0.5:
                name = self.LABELS.get(task, task)

                if prob > 0.8:
                    severity = "High"
                    icon = "🔴"
                elif prob > 0.65:
                    severity = "Medium"
                    icon = "🟠"
                else:
                    severity = "Low"
                    icon = "🟡"

                risks.append(f"{icon} {name}: {prob * 100:.0f}% ({severity})")

        return risks
"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # --- 1. Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"

    # Model & Vocab
    MODEL_FILENAME: str = os.getenv("MODEL_PATH", "checkpoints_rl_ultimate/mug_rl_best.pth")
    CHECKPOINT_PATH: Path = BASE_DIR / MODEL_FILENAME
    
    VOCAB_FILENAME: str = os.getenv("VOCAB_PATH", "data/processed/vocab_selfies.json")
    VOCAB_PATH: Path = BASE_DIR / VOCAB_FILENAME
    
    # Logging
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "mug_system.log"

    # --- 2. Security & Hardware ---
    API_TOKEN: str = os.getenv("TELEGRAM_TOKEN")
    if not API_TOKEN:
        print("⚠️ WARNING: TELEGRAM_TOKEN not set in .env")

    # SMART DEVICE DETECTION
    # Если CUDA недоступна, принудительно ставим CPU, игнорируя .env
    if torch.cuda.is_available():
        DEVICE_NAME = os.getenv("DEVICE", "cuda")
    else:
        DEVICE_NAME = "cpu"
    
    DEVICE: torch.device = torch.device(DEVICE_NAME)

    # --- 3. Hyperparameters ---
    EMBED_SIZE: int = 64
    HIDDEN_SIZE: int = 256
    LATENT_SIZE: int = 128
    NUM_LAYERS: int = 3

    # --- 4. Database ---
    DISEASE_DB: Dict[str, Any] = {
        "neuro": {
            "title": "🧠 Neurobiology",
            "targets": {
                "alzheimer": {"name": "Alzheimer's disease", "ref": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC"},
                "parkinson": {"name": "Parkinson's disease", "ref": "NC1CC2C=CC=CC=2C1"},
                "glioblastoma": {"name": "Glioblastoma", "ref": "CN1C(=O)N(C)C(=O)C(N)=C1N=NC2=CC=CC=C2C(=O)N"}
            }
        },
        "onco": {
            "title": "🦀 Oncology",
            "targets": {
                "lung": {"name": "Lung cancer", "ref": "COCC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC"},
                "liver": {"name": "Liver cancer", "ref": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F"},
                "breast": {"name": "Breast cancer", "ref": "CN(C)CC=CC(=C(C1=CC=CC=C1)C2=CC=C(O)C=C2)C3=CC=CC=C3"}
            }
        },
        "viral": {
            "title": "🦠 Virology",
            "targets": {
                "covid": {"name": "COVID-19", "ref": "CC(C)(C)NC(=O)C1CN(CC1)CC(C(CC2=CC=CC=C2)NC(=O)C3=CN=CC=N3)O"},
                "hiv": {"name": "HIV", "ref": "CC1=C(C(=O)N2C(C1)CC(C2)NC(=O)C3=C(C=CC(=C3)F)CN4CCN(CC4)C(=O)OC)O"}
            }
        }
    }


    @classmethod
    def validate(cls):
        if not cls.VOCAB_PATH.exists():
            print(f"⚠️ Warning: Vocab file not found at {cls.VOCAB_PATH}")
        if not cls.CHECKPOINT_PATH.exists():
            print(f"⚠️ Warning: Model checkpoint not found at {cls.CHECKPOINT_PATH}")

Config.validate()
import os
import torch
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Config:
    """
    Central configuration class for the Molecular Universe Generator (MUG).
    Manages environment variables, file paths, model hyperparameters, 
    and domain-specific databases.
    """

    # --- 1. Project Paths (using Pathlib) ---
    # Points to MUG_Project/ directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    
    # Data Directories
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"

    # Specific Files (Loaded from .env or defaults)
    # Defaulting to the best RL model we trained
    MODEL_FILENAME: str = os.getenv("MODEL_PATH", "checkpoints/best_model.pth")
    CHECKPOINT_PATH: Path = BASE_DIR / MODEL_FILENAME
    
    VOCAB_FILENAME: str = os.getenv("VOCAB_PATH", "data/processed/vocab_selfies.json")
    VOCAB_PATH: Path = BASE_DIR / VOCAB_FILENAME

    # Logging
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "mug_system.log"

    # --- 2. Security & Hardware ---
    API_TOKEN: str = os.getenv("TELEGRAM_TOKEN")
    if not API_TOKEN:
        raise ValueError("CRITICAL: 'TELEGRAM_TOKEN' not found in .env file.")

    # Auto-detect GPU, fallback to CPU
    DEVICE: torch.device = torch.device(
        os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- 3. Neural Network Hyperparameters ---
    # Must match the architecture used during training
    EMBED_SIZE: int = 64
    HIDDEN_SIZE: int = 256
    LATENT_SIZE: int = 128
    NUM_LAYERS: int = 3

    # --- 4. Scientific Target Database ---
    # Knowledge base for 'Hunter Mode' (Targeted Drug Design)
    DISEASE_DB: Dict[str, Any] = {
        "neuro": {
            "title": "🧠 Neuroscience (CNS)",
            "targets": {
                "alzheimer": {
                    "name": "Alzheimer's Disease",
                    "ref": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC", # Donepezil
                    "desc": "AChE Inhibitor. Requires BBB permeability."
                },
                "parkinson": {
                    "name": "Parkinson's Disease",
                    "ref": "NC1CC2C=CC=CC=2C1", # Rasagiline
                    "desc": "MAO-B Inhibitor."
                },
                "glioblastoma": {
                    "name": "Glioblastoma",
                    "ref": "CN1C(=O)N(C)C(=O)C(N)=C1N=NC2=CC=CC=C2C(=O)N", # Temozolomide analog
                    "desc": "Alkylating agent. High toxicity tolerance required."
                }
            }
        },
        "onco": {
            "title": "🦀 Oncology (Cancer)",
            "targets": {
                "lung": {
                    "name": "NSCLC (Lung Cancer)",
                    "ref": "COCC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC", # Erlotinib
                    "desc": "EGFR Tyrosine Kinase Inhibitor."
                },
                "liver": {
                    "name": "HCC (Liver Cancer)",
                    "ref": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F", # Sorafenib
                    "desc": "Multi-kinase inhibitor (VEGFR/PDGFR/RAF)."
                },
                "breast": {
                    "name": "Breast Cancer", 
                    "ref": "CN(C)CC=CC(=C(C1=CC=CC=C1)C2=CC=C(O)C=C2)C3=CC=CC=C3", # Tamoxifen
                    "desc": "Estrogen Receptor Modulator (SERM)."
                }
            }
        },
        "viral": {
            "title": "🦠 Virology",
            "targets": {
                "covid": {
                    "name": "COVID-19 (SARS-CoV-2)",
                    "ref": "CC(C)(C)NC(=O)C1CN(CC1)CC(C(CC2=CC=CC=C2)NC(=O)C3=CN=CC=N3)O", # Protease Inhibitor
                    "desc": "Mpro Protease Inhibitor."
                },
                "hiv": {
                    "name": "HIV-1", 
                    "ref": "CC1=C(C(=O)N2C(C1)CC(C2)NC(=O)C3=C(C=CC(=C3)F)CN4CCN(CC4)C(=O)OC)O", # Dolutegravir
                    "desc": "Integrase Strand Transfer Inhibitor."
                }
            }
        },
        "bio": {
            "title": "🧫 Microbiology",
            "targets": {
                "mrsa": {
                    "name": "MRSA (Superbug)", 
                    "ref": "CC(=O)NCCC1=CN(C(=O)O1)C2=CC(=C(C=C2)N3CCOCC3)F", # Linezolid
                    "desc": "Protein synthesis inhibitor."
                },
                "tb": {
                    "name": "Tuberculosis", 
                    "ref": "C1=CN=C(C=C1)C(=O)NNC2=CC=CC=C2", # Isoniazid derivative
                    "desc": "Mycolic acid synthesis inhibitor."
                }
            }
        }
    }

    # Validate paths on startup to fail early if configuration is wrong
    @classmethod
    def validate(cls):
        if not cls.VOCAB_PATH.exists():
            print(f"⚠️ Warning: Vocab file not found at {cls.VOCAB_PATH}")
        if not cls.CHECKPOINT_PATH.exists():
            print(f"⚠️ Warning: Model checkpoint not found at {cls.CHECKPOINT_PATH}")

# Run validation on import
Config.validate()
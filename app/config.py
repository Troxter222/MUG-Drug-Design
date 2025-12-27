"""
Molecular Universe Generator (MUG) - Configuration Module
Author: Ali (Troxter222)
License: MIT

Central configuration for model parameters, paths, and therapeutic targets.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Global configuration for MUG system."""
    
    # ========================
    # Directory Structure
    # ========================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"
    LOG_DIR: Path = BASE_DIR / "logs"
    CACHE_DIR: Path = BASE_DIR / "cache"
    
    # ========================
    # Model Paths
    # ========================
    MODEL_FILENAME: str = "checkpoints_rl_transformer/mug_transformer_rl_best.pth"
    CHECKPOINT_PATH: Path = BASE_DIR / MODEL_FILENAME
    
    VOCAB_FILENAME: str = "dataset/processed/vocab_transformer.json"
    VOCAB_PATH: Path = BASE_DIR / VOCAB_FILENAME
    
    LOG_FILE: Path = LOG_DIR / "mug_system.log"
    
    # ========================
    # API Configuration
    # ========================
    API_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    
    # ========================
    # Compute Device
    # ========================
    DEVICE_NAME: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    DEVICE: torch.device = torch.device(DEVICE_NAME)
    
    # ========================
    # Transformer Architecture
    # ========================
    EMBED_SIZE: int = int(os.getenv("EMBED_SIZE", "256"))
    HIDDEN_SIZE: int = int(os.getenv("HIDDEN_SIZE", "1024"))
    LATENT_SIZE: int = int(os.getenv("LATENT_SIZE", "128"))
    NUM_LAYERS: int = int(os.getenv("NUM_LAYERS", "4"))
    NHEAD: int = int(os.getenv("NHEAD", "8"))
    
    # ========================
    # Generation Parameters
    # ========================
    MAX_SEQUENCE_LENGTH: int = 200
    SAMPLING_TEMPERATURE: float = 1.0
    RANDOM_BATCH_SIZE: int = 10
    TARGETED_BATCH_SIZE: int = 50
    MAX_GENERATION_ATTEMPTS: int = 10
    
    # ========================
    # Molecular Filters
    # ========================
    MIN_MOLECULAR_WEIGHT: float = 100.0
    MAX_MOLECULAR_WEIGHT: float = 900.0
    MIN_QED_THRESHOLD: float = 0.1
    TARGETED_QED_THRESHOLD: float = 0.2

    # ========================
    # Model Registry & Paths
    # ========================

    MODEL_REGISTRY: Dict[str, Any] = {
        "rl_neuro_v2": {
            "name": "🧠 Neuro-Hunter V2 (RL Optimized 'BEST')",
            "path": "checkpoints_rl_finetuned/mug_rl_neuro_v2.pth",
            "type": "transformer",
            "vocab": "dataset/processed/vocab_transformer.json",
            "params": {"embed": 256, "hidden": 1024, "layers": 4, "nhead": 8}
        },
        "trans_rl_best": {
            "name": "🚀 Transformer RL",
            "path": "checkpoints_rl_transformer/mug_transformer_rl_best.pth",
            "type": "transformer",
            "vocab": "dataset/processed/vocab_transformer.json",
            "params": {"embed": 256, "hidden": 1024, "layers": 4, "nhead": 8}
        },
        "trans_v2": {
            "name": "🔹 Transformer V2 (Epoch 11)",
            "path": "checkpoints_transformer_v2/mug_trans_v2_ep11.pth",
            "type": "transformer",
            "vocab": "dataset/processed/vocab_transformer.json",
            "params": {"embed": 256, "hidden": 1024, "layers": 4, "nhead": 8}
        },
        "mug_universal_v1": {
            "name": "🔹 Mug Universal",
            "path": "checkpoints_rl_universal/mug_universal_v1.pth",
            "type": "transformer",
            "vocab": "dataset/processed/vocab_transformer.json",
            "params": {"embed": 256, "hidden": 1024, "layers": 4, "nhead": 8}
        },
        "gru_base": {
            "name": "🕰️ GRU Base (Legacy)",
            "path": "data/checkpoints_selfies/mug_selfies_epoch_10.pth",
            "type": "gru",
            "vocab": "dataset/processed/vocab_transformer.json",
            "params": {"embed": 64, "hidden": 256, "layers": 3, "latent": 128} 
        },
        "gru_rl": {
            "name": "🧪 GRU RL (Legacy)",
            "path": "data/checkpoints_rl/mug_rl_best.pth",
            "type": "gru",
            "vocab": "dataset/processed/vocab_transformer.json",
            "params": {"embed": 64, "hidden": 256, "layers": 3, "latent": 128}
        },
    }

    # По умолчанию берем Transformer RL
    CURRENT_MODEL_KEY = "rl_neuro_v2"
    
    # Эти пути теперь динамические, но оставим как fallback
    MODEL_FILENAME: str = MODEL_REGISTRY[CURRENT_MODEL_KEY]["path"]
    CHECKPOINT_PATH: Path = BASE_DIR / MODEL_FILENAME
    VOCAB_FILENAME: str = MODEL_REGISTRY[CURRENT_MODEL_KEY]["vocab"]
    VOCAB_PATH: Path = BASE_DIR / VOCAB_FILENAME
    
    # ========================
    # Therapeutic Target Database
    # ========================
    DISEASE_DB: Dict[str, Any] = {
        "neuro": {
            "title": "🧠 Neuroscience",
            "targets": {
                "alzheimer": {
                    "disease": "Alzheimer's Disease",
                    "target_name": "Acetylcholinesterase (AChE)",
                    "target_class": "Enzyme (Hydrolase)",
                    "ref": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC", # Donepezil
                    "pdb_id": "1EVE"
                },
                "schizophrenia": {
                    "disease": "Schizophrenia",
                    "target_name": "Dopamine D2 Receptor (DRD2)",
                    "target_class": "GPCR (Class A)",
                    "ref": "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=C2C=CC(=C4)Cl", # Clozapine derivative
                    "pdb_id": "6CM4"
                }
            }
        },
        "onco": {
            "title": "🦀 Oncology",
            "targets": {
                "lung": {
                    "disease": "NSCLC (Lung Cancer)",
                    "target_name": "EGFR Kinase (L858R)",
                    "target_class": "Kinase (Tyrosine)",
                    "ref": "COCC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC", # Erlotinib
                    "pdb_id": "1M17"
                },
                "melanoma": {
                    "disease": "Melanoma",
                    "target_name": "BRAF (V600E)",
                    "target_class": "Kinase (Serine/Threonine)",
                    "ref": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC=C(C=C2)Cl", # Vemurafenib analog
                    "pdb_id": "3OG7"
                }
            }
        },
        "viral": {
            "title": "🦠 Virology",
            "targets": {
                "covid": {
                    "disease": "COVID-19",
                    "target_name": "SARS-CoV-2 M-pro (3CL)",
                    "target_class": "Protease (Cysteine)",
                    "ref": "CC(C)(C)NC(=O)C1CN(CC1)CC(C(CC2=CC=CC=C2)NC(=O)C3=CN=CC=N3)O", # Paxlovid component
                    "pdb_id": "6LU7"
                },
                "hiv": {
                    "disease": "HIV-1 Infection",
                    "target_name": "HIV-1 Protease",
                    "target_class": "Protease (Aspartyl)",
                    "ref": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2CCOC2)O)S(=O)(=O)C3=CC=C(C=C3)N", # Darunavir
                    "pdb_id": "1HSG"
                }
            }
        }
    }
    
    # ========================
    # Validation & Initialization
    # ========================
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration and check for required files.
        
        Returns:
            True if all validations pass, False otherwise
        """
        is_valid = True
        
        # Check API token
        if not cls.API_TOKEN:
            warnings.warn("⚠️ TELEGRAM_TOKEN not found in environment", UserWarning)
            is_valid = False
        
        # Check vocabulary file
        if not cls.VOCAB_PATH.exists():
            warnings.warn(f"⚠️ Vocabulary file missing: {cls.VOCAB_PATH}", UserWarning)
            is_valid = False
        
        # Check model checkpoint
        if not cls.CHECKPOINT_PATH.exists():
            warnings.warn(f"⚠️ Model checkpoint missing: {cls.CHECKPOINT_PATH}", UserWarning)
            is_valid = False
        
        # Create necessary directories
        for directory in [cls.LOG_DIR, cls.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate device
        if cls.DEVICE_NAME == "cuda" and not torch.cuda.is_available():
            warnings.warn("⚠️ CUDA requested but not available. Falling back to CPU.", UserWarning)
            cls.DEVICE = torch.device("cpu")
        
        return is_valid
    
    @classmethod
    def get_target_info(cls, category: str, target: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve target information by category and key.
        
        Args:
            category: Disease category key
            target: Target key within category
            
        Returns:
            Target information dictionary or None if not found
        """
        try:
            return cls.DISEASE_DB[category]['targets'][target]
        except KeyError:
            return None
    
    @classmethod
    def get_all_categories(cls) -> Dict[str, str]:
        """
        Get all available therapeutic categories.
        
        Returns:
            Dictionary mapping category keys to titles
        """
        return {key: value['title'] for key, value in cls.DISEASE_DB.items()}
    
    @classmethod
    def get_category_targets(cls, category: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all targets within a category.
        
        Args:
            category: Disease category key
            
        Returns:
            Dictionary of targets
        """
        return cls.DISEASE_DB.get(category, {}).get('targets', {})
    
    @classmethod
    def print_summary(cls):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("MUG Configuration Summary")
        print("="*60)
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.CHECKPOINT_PATH.name}")
        print(f"Vocabulary: {cls.VOCAB_PATH.name}")
        print(f"Architecture: d_model={cls.EMBED_SIZE}, layers={cls.NUM_LAYERS}, heads={cls.NHEAD}")
        print(f"Therapeutic Areas: {len(cls.DISEASE_DB)}")
        total_targets = sum(len(cat['targets']) for cat in cls.DISEASE_DB.values())
        print(f"Total Targets: {total_targets}")
        print("="*60 + "\n")


# Run validation on import
Config.validate()
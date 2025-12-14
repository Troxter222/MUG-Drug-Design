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
    # Therapeutic Target Database
    # ========================
    DISEASE_DB: Dict[str, Any] = {
        "neuro": {
            "title": "🧠 Neuroscience",
            "description": "Central nervous system disorders",
            "targets": {
                "alzheimer": {
                    "name": "Alzheimer's Disease",
                    "ref": "COC1=C(C=C2C(=C1)CC(C2=O)CC3CCN(CC3)CC4=CC=CC=C4)OC",
                    "protein": "Acetylcholinesterase"
                },
                "parkinson": {
                    "name": "Parkinson's Disease",
                    "ref": "NC1CC2C=CC=CC=2C1",
                    "protein": "MAO-B"
                },
                "glioblastoma": {
                    "name": "Glioblastoma",
                    "ref": "CN1C(=O)N(C)C(=O)C(N)=C1N=NC2=CC=CC=C2C(=O)N",
                    "protein": "EGFR"
                },
                "epilepsy": {
                    "name": "Epilepsy",
                    "ref": "NC(=O)C1=CC=C(C=C1)N2C(=O)CN=C2",
                    "protein": "GABA-A Receptor"
                },
                "depression": {
                    "name": "Major Depression",
                    "ref": "CNCCC(C1=CC=CC=C1)OC2=CC=CC3=CC=CC=C32",
                    "protein": "Serotonin Transporter"
                },
                "schizophrenia": {
                    "name": "Schizophrenia",
                    "ref": "CN1CCN(CC1)C2=NC3=CC=CC=C3NC4=C2C=CC(=C4)Cl",
                    "protein": "Dopamine D2 Receptor"
                }
            }
        },
        
        "onco": {
            "title": "🦀 Oncology",
            "description": "Cancer therapeutics",
            "targets": {
                "lung": {
                    "name": "Lung Cancer (NSCLC)",
                    "ref": "COCC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC(=C3)C#C)OCCOC",
                    "protein": "EGFR"
                },
                "liver": {
                    "name": "Hepatocellular Carcinoma",
                    "ref": "CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F",
                    "protein": "VEGFR"
                },
                "breast": {
                    "name": "Breast Cancer",
                    "ref": "CCC(=O)NC1=CC=C(C=C1)N(CCCl)CCCl",
                    "protein": "Estrogen Receptor"
                },
                "prostate": {
                    "name": "Prostate Cancer",
                    "ref": "CNC(=O)C1=C(SC=C1)OCC2=CC=C(C=C2)F",
                    "protein": "Androgen Receptor"
                },
                "melanoma": {
                    "name": "Melanoma",
                    "ref": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC=C(C=C2)Cl",
                    "protein": "BRAF V600E"
                },
                "leukemia": {
                    "name": "Chronic Myeloid Leukemia",
                    "ref": "CN1CCN(CC1)CC2=CC=C(C=C2)NC(=O)C3=CC(=C(C=C3)NC4=NC=CC(=N4)C5=CN=CC=C5)C(F)(F)F",
                    "protein": "BCR-ABL"
                }
            }
        },
        
        "viral": {
            "title": "🦠 Infectious Diseases",
            "description": "Viral and bacterial infections",
            "targets": {
                "covid": {
                    "name": "COVID-19",
                    "ref": "CC(C)(C)NC(=O)C1CN(CC1)CC(C(CC2=CC=CC=C2)NC(=O)C3=CN=CC=N3)O",
                    "protein": "3CL Protease"
                },
                "influenza": {
                    "name": "Influenza",
                    "ref": "CCC(CC)OC1C=C(CC(C1NC(=O)C)N)C(=O)O",
                    "protein": "Neuraminidase"
                },
                "hiv": {
                    "name": "HIV/AIDS",
                    "ref": "CC(C)CN(CC(C(CC1=CC=CC=C1)NC(=O)OC2CCOC2)O)S(=O)(=O)C3=CC=C(C=C3)N",
                    "protein": "HIV Protease"
                },
                "hepatitis_c": {
                    "name": "Hepatitis C",
                    "ref": "CC(C)(C)NC(=O)C1CN(C2CC3(CC2C1)NC(=O)NC3=O)C(=O)C(NC(=O)OC)C(C)(C)C",
                    "protein": "NS3/4A Protease"
                },
                "tuberculosis": {
                    "name": "Tuberculosis",
                    "ref": "CC1=NC=C(C(=C1)C(=O)NC2=CC=C(C=C2)N3CCOCC3)C",
                    "protein": "InhA"
                }
            }
        },
        
        "cardio": {
            "title": "❤️ Cardiovascular",
            "description": "Heart and vascular disorders",
            "targets": {
                "hypertension": {
                    "name": "Hypertension",
                    "ref": "CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
                    "protein": "Angiotensin II Receptor"
                },
                "thrombosis": {
                    "name": "Thrombosis",
                    "ref": "CCC(=O)N1CCCC1C(=O)N2C3CCCCC3CC2C(=O)O",
                    "protein": "Factor Xa"
                },
                "heart_failure": {
                    "name": "Heart Failure",
                    "ref": "CCCCC(CC)COC(=O)C(C)NP(=O)(OC1=CC=CC=C1)OC2=CC=CC=C2",
                    "protein": "ACE"
                },
                "arrhythmia": {
                    "name": "Cardiac Arrhythmia",
                    "ref": "CCCCNC1=C2C(=NC(=N1)N)N(C=N2)C3C(C(C(O3)COP(=O)(O)O)O)O",
                    "protein": "hERG Channel"
                }
            }
        },
        
        "metabolic": {
            "title": "⚡ Metabolic Disorders",
            "description": "Diabetes and metabolic syndrome",
            "targets": {
                "diabetes_t2": {
                    "name": "Type 2 Diabetes",
                    "ref": "CC(C)C(CC1=CC(=C(C=C1)O)O)NC(C2CCC(CC2)(C#N)N)O",
                    "protein": "DPP-4"
                },
                "obesity": {
                    "name": "Obesity",
                    "ref": "CN1C2CCC1CC(C2)OC(=O)C(CO)C3=CC=CC=C3",
                    "protein": "MC4 Receptor"
                },
                "hyperlipidemia": {
                    "name": "Hyperlipidemia",
                    "ref": "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",
                    "protein": "HMG-CoA Reductase"
                }
            }
        },
        
        "immune": {
            "title": "🛡️ Immunology",
            "description": "Autoimmune and inflammatory diseases",
            "targets": {
                "rheumatoid": {
                    "name": "Rheumatoid Arthritis",
                    "ref": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
                    "protein": "JAK"
                },
                "psoriasis": {
                    "name": "Psoriasis",
                    "ref": "CN1C2=C(C=C(C=C2)N)NC1=O",
                    "protein": "PDE4"
                },
                "lupus": {
                    "name": "Systemic Lupus",
                    "ref": "C1CN(CCN1)C2=CC=C(C=C2)NC(=O)C3=CC4=C(C=C3)N=CN=C4N",
                    "protein": "TLR7/9"
                },
                "crohns": {
                    "name": "Crohn's Disease",
                    "ref": "CC1=CC=C(C=C1)C2=C(N=C(S2)N3CCN(CC3)C)C4=CC=C(C=C4)F",
                    "protein": "TNF-α"
                }
            }
        },
        
        "pain": {
            "title": "💊 Pain Management",
            "description": "Analgesics and pain relief",
            "targets": {
                "chronic_pain": {
                    "name": "Chronic Pain",
                    "ref": "CN1CCC(CC1)N2C3=CC=CC=C3SC4=C2C=C(C=C4)Cl",
                    "protein": "μ-Opioid Receptor"
                },
                "neuropathic": {
                    "name": "Neuropathic Pain",
                    "ref": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                    "protein": "COX-2"
                },
                "migraine": {
                    "name": "Migraine",
                    "ref": "CN1C2=CC=CC=C2C(=O)N(C1=O)CC3CCCCC3",
                    "protein": "5-HT1B Receptor"
                }
            }
        },
        
        "respiratory": {
            "title": "🫁 Respiratory",
            "description": "Pulmonary disorders",
            "targets": {
                "asthma": {
                    "name": "Asthma",
                    "ref": "CC(C)(C)NCC(COC1=CC=CC2=C1CCCC2)O",
                    "protein": "β2-Adrenergic Receptor"
                },
                "copd": {
                    "name": "COPD",
                    "ref": "CN(C)C(=O)OC1=CC=CC(=C1)C(C)N",
                    "protein": "Muscarinic Receptor"
                },
                "fibrosis": {
                    "name": "Pulmonary Fibrosis",
                    "ref": "COC1=CC2=C(C=C1)C(=CN=C2NC3=CC(=C(C=C3)F)Cl)C#C",
                    "protein": "TGF-β Receptor"
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
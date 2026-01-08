import sys
import os
from pathlib import Path

# Add the project directory to sys.path
sys.path.append(os.getcwd())

from app.config import Config

def verify_gnn_tox():
    print("Verifying gnn_tox configuration...")
    
    # Get the gnn_tox config
    model_config = Config.MODEL_REGISTRY.get('gnn_tox')
    if not model_config:
        print("ERROR: 'gnn_tox' key not found in MODEL_REGISTRY")
        return False
        
    print(f"Found config: {model_config}")
    
    # Check Model Path
    model_path = Config.BASE_DIR / model_config['path']
    if model_path.exists():
        print(f"SUCCESS: Model file found at {model_path}")
    else:
        print(f"FAILURE: Model file NOT found at {model_path}")
        return False
        
    # Check Vocab Path
    vocab_path = Config.BASE_DIR / model_config['vocab']
    if vocab_path.exists():
        print(f"SUCCESS: Vocab file found at {vocab_path}")
    else:
        print(f"FAILURE: Vocab file NOT found at {vocab_path}")
        return False
        
    print("Verification Passed!")
    return True

if __name__ == "__main__":
    if verify_gnn_tox():
        sys.exit(0)
    else:
        sys.exit(1)

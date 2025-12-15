import os
import requests
import json
from pathlib import Path

# Setup paths relative to script location
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RECEPTOR_DIR = BASE_DIR / "data" / "receptors"
RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "alzheimer": {"pdb": "1EVE", "center": [2.3, 64.6, 67.4], "size": [22, 22, 22]},
    "lung":      {"pdb": "1M17", "center": [22.0, 0.0, 5.0],  "size": [20, 20, 20]},
    "covid":     {"pdb": "6LU7", "center": [-10.7, 12.5, 68.4], "size": [24, 24, 24]},
    "hiv":       {"pdb": "1HSG", "center": [9.0, 9.0, 9.0], "size": [22, 22, 22]},
}

def setup():
    print(f"🧬 Setting up receptors in {RECEPTOR_DIR}...")
    
    config_data = {}
    
    for key, data in TARGETS.items():
        pdb_file = RECEPTOR_DIR / f"{key}.pdb"
        
        # Download if missing
        if not pdb_file.exists():
            print(f"⬇️ Downloading {data['pdb']}...")
            url = f"https://files.rcsb.org/download/{data['pdb']}.pdb"
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    with open(pdb_file, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"❌ Failed to download {data['pdb']}")
                    continue
            except Exception as e:
                print(f"❌ Network error: {e}")
                continue
        
        # Save config
        config_data[key] = {
            "pdb_file": str(pdb_file.absolute()),
            "center": data['center'],
            "size": data['size']
        }

    # Write config file
    with open(RECEPTOR_DIR / "targets_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
        
    print("✅ Receptor setup complete.")

if __name__ == "__main__":
    setup()
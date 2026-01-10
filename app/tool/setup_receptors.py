"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import requests
import json
from pathlib import Path
from rdkit import Chem
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ReceptorSetup")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RECEPTOR_DIR = BASE_DIR / "data" / "receptors"
RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)

# Target database
TARGETS = {
    "alzheimer": {
        "pdb_id": "1EVE", 
        "center": [2.3, 64.6, 67.4], 
        "size": [20, 20, 20],
        "name": "Acetylcholinesterase"
    },
    "lung": {
        "pdb_id": "1M17", # EGFR
        "center": [22.0, 0.0, 5.0], 
        "size": [20, 20, 20],
        "name": "EGFR Kinase"
    },
    "covid": {
        "pdb_id": "6LU7", # SARS-CoV-2 M-pro
        "center": [-10.7, 12.5, 68.4], 
        "size": [20, 20, 20],
        "name": "M-pro"
    }
}

def clean_pdb_with_rdkit(input_pdb, output_pdb, chain_to_keep="A"):
    """
    Removes extra chains and heteroatoms, keeping the structure intact.
    """
    try:
        # Load original
        mol = Chem.MolFromPDBFile(str(input_pdb), sanitize=False, removeHs=True)
        if not mol:
            logger.error(f"Could not parse {input_pdb}")
            return False

        # Use RWMol to edit the existing structure
        edit_mol = Chem.RWMol(mol)
        atoms_to_remove = []

        for atom in edit_mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None:
                atoms_to_remove.append(atom.GetIdx())
                continue
                
            # If atom is not from the desired chain OR is a heteroatom (ligand/water) - mark for removal
            if info.GetChainId().strip() != chain_to_keep or info.GetIsHeteroAtom():
                atoms_to_remove.append(atom.GetIdx())

        # Remove atoms from the end (to prevent index shifting)
        for idx in sorted(atoms_to_remove, reverse=True):
            edit_mol.RemoveAtom(idx)

        # Save result
        with open(output_pdb, 'w') as f:
            f.write(Chem.MolToPDBBlock(edit_mol))
        
        logger.info(f"Successfully cleaned. Atoms kept: {edit_mol.GetNumAtoms()}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning PDB: {e}")
        return False

def setup():
    logger.info(f"Starting Receptor Pipeline in {RECEPTOR_DIR}...")
    
    config_data = {}
    
    for key, data in TARGETS.items():
        raw_pdb = RECEPTOR_DIR / f"{key}_raw.pdb"
        clean_pdb = RECEPTOR_DIR / f"{key}_clean.pdb"
        pdbqt_file = RECEPTOR_DIR / f"{key}.pdbqt"
        
        # 1. Downloading
        if not raw_pdb.exists():
            logger.info(f"⬇Downloading {data['pdb_id']}...")
            url = f"https://files.rcsb.org/download/{data['pdb_id']}.pdb"
            r = requests.get(url, timeout=30)  # nosec
            with open(raw_pdb, "wb") as f:
                f.write(r.content)
        
        # 2. Cleaning via RDKit
        if not clean_pdb.exists():
            logger.info(f"Cleaning {key} (Chain A only, no water/ligands)...")
            success = clean_pdb_with_rdkit(raw_pdb, clean_pdb)
            if not success:
                logger.error(f"Failed to clean {key}")
                continue
        
        # 3. PDBQT Preparation
        # Ideally, Meeko or ADFRTools should be used here. 
        # For now, we use a smart placeholder if files are missing.
        if not pdbqt_file.exists():
            logger.warning(f"{key}.pdbqt missing. Please use 'obabel' or MGLTools to convert {key}_clean.pdb")
            # Copy for testing purposes, but log warning
            with open(clean_pdb, 'rb') as src, open(pdbqt_file, 'wb') as dst:
                dst.write(src.read())

        config_data[key] = {
            "receptor_path": str(pdbqt_file.absolute()),
            "center": data['center'],
            "size": data['size'],
            "name": data['name']
        }

    with open(RECEPTOR_DIR / "targets_config.json", "w") as f:
        json.dump(config_data, f, indent=2)
        
    logger.info("Receptor configuration updated.")

if __name__ == "__main__":
    setup()
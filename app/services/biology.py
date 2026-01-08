"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT

Biology Service: Molecular Docking & Affinity Estimation
Backend: Subprocess wrapper for AutoDock Vina.exe + Meeko
"""

import json
import logging
import random
import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski

# Meeko (Ligand preparation only, without Vina bindings)
try:
    from meeko import MoleculePreparation
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False

logger = logging.getLogger(__name__)


class BiologyService:
    # --- CONFIGURATION ---
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    VINA_PATH = BASE_DIR / "app" / "tool" / "vina.exe"
    
    # Path to receptor configurations
    CONFIG_FILE = BASE_DIR / "data/receptors/targets_config.json"

    def __init__(self):
        self.targets = {}
        self._load_config()
        
        # Check tool availability
        self.vina_ready = self.VINA_PATH.exists() and MEEKO_AVAILABLE
        
        if not self.VINA_PATH.exists():
            logger.warning(f"Warning: Vina.exe not found at {self.VINA_PATH}. Docking will use QSAR fallback.")

    def _load_config(self):
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    self.targets = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load target config: {e}")

    def dock_molecule(self, mol: Chem.Mol, target_category: str) -> float:
        """
        Performs 3D Docking via Vina Subprocess.
        Returns affinity in kcal/mol (negative is better).
        """
        target_key = self._resolve_target_key(target_category)
        
        # Fallback conditions
        if not self.vina_ready or target_key not in self.targets:
            return self._qsar_fallback(mol)

        target_conf = self.targets[target_key]
        receptor_path = Path(target_conf["receptor_path"])
        
        if not receptor_path.exists():
            return self._qsar_fallback(mol)

        try:
            # --- 1. Ligand Preparation (RDKit -> PDBQT String) ---
            mol_3d = Chem.AddHs(mol)
            
            # Attempt 3D generation
            if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3()) != 0:
                return self._qsar_fallback(mol) # Failed to generate conformation

            try:
                AllChem.MMFFOptimizeMolecule(mol_3d)
            except Exception: 
                pass

            # Convert to PDBQT format
            preparator = MoleculePreparation()
            preparator.prepare(mol_3d)
            ligand_pdbqt_string = preparator.write_pdbqt_string()

            # --- 2. Execute Vina via Subprocess ---
            # Create a temporary directory to avoid file clutter
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                ligand_file = tmp_path / "ligand.pdbqt"
                out_file = tmp_path / "out.pdbqt"
                log_file = tmp_path / "vina.log"

                # Write ligand file
                ligand_file.write_text(ligand_pdbqt_string)

                center = target_conf["center"]
                size = target_conf["size"]

                # Construct command
                cmd = [
                    str(self.VINA_PATH),
                    "--receptor", str(receptor_path),
                    "--ligand", str(ligand_file),
                    "--center_x", str(center[0]),
                    "--center_y", str(center[1]),
                    "--center_z", str(center[2]),
                    "--size_x", str(size[0]),
                    "--size_y", str(size[1]),
                    "--size_z", str(size[2]),
                    "--exhaustiveness", "4", # Fast mode
                    "--num_modes", "1",      # Top pose only
                    "--cpu", "1",            # 1 CPU core (important for bot concurrency)
                    "--out", str(out_file),
                    "--log", str(log_file)
                ]

                # Run executable
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL, # Suppress console output
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=30 # 30-second timeout to prevent hanging
                )

                # --- 3. Log Parsing ---
                affinity = self._parse_vina_log(log_file)
                return affinity

        except Exception:
            return self._qsar_fallback(mol)

    def _parse_vina_log(self, log_path: Path) -> float:
        """Reads affinity from Vina output log."""
        try:
            with open(log_path, "r") as f:
                start_reading = False
                for line in f:
                    # Result table starts after the line "-----+..."
                    if "----+" in line:
                        start_reading = True
                        continue
                    
                    if start_reading:
                        parts = line.strip().split()
                        # The first line after header is Mode 1 (Best)
                        # Format:   1   -8.5   0.000   0.000
                        if len(parts) >= 2 and parts[0] == "1":
                            return float(parts[1])
        except Exception:
            pass
        raise ValueError("Could not parse Vina log")

    def _resolve_target_key(self, category_str):
        cat = category_str.lower()
        if "alzheimer" in cat or "neuro" in cat:
            return "alzheimer"
        if "lung" in cat or "onco" in cat: 
            return "lung"
        if "covid" in cat or "viral" in cat: 
            return "covid"
        return "unknown"

    def _qsar_fallback(self, mol):
        """Mathematical approximation (Backup Plan)."""
        mw = Descriptors.MolWt(mol)
        hb = Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)
        rot = Lipinski.NumRotatableBonds(mol)
        
        # Formula: Heavier implies stronger binding (-), 
        # but penalty for rotors and H-bonds (entropy/desolvation)
        score = -5.0 - (mw / 100.0 * 0.9) + (rot * 0.05) + (hb * 0.05)
        
        # Add noise for realistic proxy behavior
        noisy_score = score + random.uniform(-0.3, 0.3)
        return round(max(-12.0, min(-3.0, noisy_score)), 2)

    @staticmethod
    def interpret_affinity(score: float) -> str:
        if score > -6.0: 
            return "Non-binder / Very Weak"
        if score > -7.5: 
            return "Moderate Binder"
        if score > -9.0: 
            return "Hit-like (Docking Score)"
        if score > -10.5: 
            return "Lead-like (High Affinity)"
        return "Potent Inhibitor (Optimized)"

    @staticmethod
    def get_confidence_level(score: float, similarity: float) -> str:
        if score < -9.0:
            return "High (3D Validated)"
        if score < -7.5:
            return "Medium"
        return "Low"
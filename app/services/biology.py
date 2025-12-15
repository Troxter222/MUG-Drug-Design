import os
import json
import random
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem

# Configure logger
logger = logging.getLogger(__name__)

# Try importing Vina (optional dependency)
try:
    from vina import Vina
    from meeko import MoleculePreparation
    VINA_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ AutoDock Vina / Meeko not installed. Using QSAR approximation.")
    VINA_AVAILABLE = False

class BiologyService:
    """
    Bio-physics simulation service. 
    Uses AutoDock Vina if available, otherwise falls back to QSAR scoring.
    """
    
    # Path configuration
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RECEPTOR_DIR = BASE_DIR / "data" / "receptors"
    CONFIG_FILE = RECEPTOR_DIR / "targets_config.json"

    def __init__(self):
        self.config = {}
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load receptor config: {e}")

    def _get_target_config(self, target_type: str):
        """Finds receptor configuration for a given disease."""
        # Simple mapping logic
        key = target_type
        if target_type not in self.config:
            # Fallback categories
            if "neuro" in target_type:
                key = "alzheimer"
            elif "onco" in target_type:
                key = "lung"
            elif "viral" in target_type:
                key = "covid"
            elif "hiv" in target_type:
                key = "hiv"
        
        return self.config.get(key)

    def dock_molecule(self, mol: Chem.Mol, target_type: str) -> float:
        """
        Main entry point. Routes to Vina or QSAR.
        """
        # If Vina is installed and we have a protein file -> Real Docking
        target_cfg = self._get_target_config(target_type)
        
        if VINA_AVAILABLE and target_cfg and os.path.exists(target_cfg.get('pdb_file', '')):
            return self._run_vina(mol, target_cfg)
        else:
            return self._run_qsar(mol, target_type)

    def _run_vina(self, mol, target_cfg):
        """Executes AutoDock Vina simulation."""
        try:
            # 1. Prepare Ligand
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            ligand_string = preparator.write_pdbqt_string()

            # 2. Setup Vina
            v = Vina(sf_name='vina')
            v.set_receptor(target_cfg['pdb_file'])
            v.set_ligand_from_string(ligand_string)
            
            # 3. Compute
            v.compute_vina_maps(center=target_cfg['center'], box_size=target_cfg['size'])
            v.dock(exhaustiveness=4, n_poses=1) # Low exhaustiveness for speed
            
            return round(v.score()[0], 2)
            
        except Exception as e:
            logger.error(f"Vina Docking failed: {e}. Falling back to QSAR.")
            return self._run_qsar(mol, "unknown")

    def _run_qsar(self, mol, target_type):
        """Mathematical approximation of binding affinity."""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hb = Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        
        # Base binding energy formula
        score = -5.0 - (mw / 100.0) * 0.8 - (hb * 0.1) + (rotors * 0.1)
        
        # Target specific adjustments
        if "neuro" in target_type or "alzheimer" in target_type:
            if logp > 2.5: 
                score -= 1.0 # Lipophilicity bonus
        elif "viral" in target_type or "covid" in target_type:
             if hb > 6: 
                 score -= 1.0 # H-bond bonus
             
        # Stochastic noise (simulation variance)
        final_score = score + random.uniform(-0.5, 0.5)
        return round(max(-12.0, min(-3.0, final_score)), 2)

    @staticmethod
    def interpret_affinity(score):
        if score > -6.0: 
            return "❌ Weak Binding"
        if score > -8.0: 
            return "⚠️ Moderate Affinity"
        if score > -9.5: 
            return "✅ Strong Binding (Lead)"
        return "💎 **POTENT INHIBITOR**"
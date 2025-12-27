"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

from typing import List
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, Lipinski

class RetrosynthesisService:
    """
    Service responsible for breaking down complex molecules into simpler 
    synthetic building blocks (precursors) using reaction-based rules.
    """
    
    # Maximum number of fragments to return to avoid UI clutter
    MAX_FRAGMENTS_DISPLAY = 4

    @staticmethod
    def get_building_blocks(mol: Chem.Mol) -> List[str]:
        """
        Decomposes a molecule into potential synthetic precursors using the 
        BRICS (Breaking of Retrosynthetically Interesting Chemical Substructures) algorithm.
        """
        try:
            # 1. Apply BRICS rules to break strategic bonds
            fragmented_mol = BRICS.BreakBRICSBonds(mol)
            
            # 2. Convert to SMILES to separate fragments (dot-separated)
            raw_fragments = Chem.MolToSmiles(fragmented_mol).split(".")
            
            clean_precursors = []
            
            for frag_smiles in raw_fragments:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                
                if frag_mol:
                    # 3. Clean up "Dummy Atoms"
                    for atom in frag_mol.GetAtoms():
                        if atom.GetAtomicNum() == 0: 
                            atom.SetAtomicNum(1) # Cap with Hydrogen
                    
                    try:
                        Chem.SanitizeMol(frag_mol)
                        clean_precursors.append(Chem.MolToSmiles(frag_mol))
                    except Exception:
                        continue
            
            # 4. Post-processing
            unique_frags = sorted(list(set(clean_precursors)), key=len, reverse=True)
            return unique_frags[:RetrosynthesisService.MAX_FRAGMENTS_DISPLAY]
            
        except Exception:
            return []

    @staticmethod
    def assess_complexity(mol: Chem.Mol) -> str:
        """
        Detects structural complexity markers (Fused rings, high Bertz complexity).
        """
        try:
            # Считаем кольца
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            
            # Эвристика: Bertz Complexity (мера сложности графа)
            complexity = Descriptors.BertzCT(mol)
            
            if complexity > 800 or num_rings > 4:
                return "High (Fused/Complex System)"
            elif complexity > 400:
                return "Medium"
            return "Low"
        except Exception:
            return "Unknown"

    @staticmethod
    def describe_synthesis(fragments: List[str], complexity_level: str = "Low") -> str:
        """
        Scientific assessment of synthetic feasibility based on fragments and complexity.
        """
        if not fragments:
            return "🔹 **One-pot potential / Monolithic core** (Complex scaffold)"
        
        count = len(fragments)
        
        # Базовая логика по количеству кусков
        if count <= 2:
            strategy = "Simple Derivatization / Coupling"
            feasibility = "High"
        elif count <= 4:
            strategy = "Convergent Assembly"
            feasibility = "Moderate"
        else:
            strategy = "Multi-step Linear Synthesis"
            feasibility = "Low (High Complexity)"

        # PENALTY RULE: Если структура сложная (Fused/Bridged), синтез трудный
        # даже если фрагментов мало (например, сложный полицикл)
        if complexity_level == "High (Fused/Complex System)":
            feasibility = "Low (Complex Scaffold)"
        elif complexity_level == "Medium" and feasibility == "High":
             feasibility = "Medium (Cyclization required)"

        desc = (f"Strategy: _{strategy}_\n"
                f"Feasibility: _{feasibility}_\n"
                f"Complexity: _{complexity_level}_\n"
                f"Key Building Blocks identified: {count}")
        
        return desc
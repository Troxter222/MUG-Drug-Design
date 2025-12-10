"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

from typing import List
from rdkit import Chem
from rdkit.Chem import BRICS

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

        Args:
            mol: The RDKit molecule object to decompose.

        Returns:
            List[str]: A list of unique SMILES strings representing the building blocks.
        """
        try:
            # 1. Apply BRICS rules to break strategic bonds
            fragmented_mol = BRICS.BreakBRICSBonds(mol)
            
            # 2. Convert to SMILES to separate fragments (dot-separated)
            # Example result: "c1ccccc1[1*].[16*]C(=O)O"
            raw_fragments = Chem.MolToSmiles(fragmented_mol).split(".")
            
            clean_precursors = []
            
            for frag_smiles in raw_fragments:
                frag_mol = Chem.MolFromSmiles(frag_smiles)
                
                if frag_mol:
                    # 3. Clean up "Dummy Atoms" (Isotopes/Wildcards from BRICS)
                    # BRICS leaves markers like [1*] at breaking points. 
                    # We convert them to Hydrogens to make valid chemical entities.
                    for atom in frag_mol.GetAtoms():
                        if atom.GetAtomicNum() == 0: 
                            atom.SetAtomicNum(1) # Cap with Hydrogen
                    
                    try:
                        # Ensure the fragment is valid chemistry
                        Chem.SanitizeMol(frag_mol)
                        clean_precursors.append(Chem.MolToSmiles(frag_mol))
                    except Exception:
                        continue
            
            # 4. Post-processing: Deduplicate and Sort
            # We prioritize larger fragments as they represent the "core" scaffolds.
            unique_frags = sorted(list(set(clean_precursors)), key=len, reverse=True)
            
            return unique_frags[:RetrosynthesisService.MAX_FRAGMENTS_DISPLAY]
            
        except Exception:
            # Retrosynthesis failures should not crash the pipeline
            return []

    @staticmethod
    def describe_synthesis(fragments: List[str]) -> str:
        """
        Generates a human-readable synthesis strategy description based on the 
        identified precursors.
        """
        if not fragments:
            return "🔹 **De Novo Synthesis** (Monolithic Structure / No obvious cuts)"
        
        # Format list items for display
        steps = [f"{i+1}️⃣ `{frag}`" for i, frag in enumerate(fragments)]
        
        if len(fragments) >= 2:
            # Multicomponent reaction or convergent synthesis
            return "🔹 **Convergent Assembly Strategy:**\n" + "\n➕\n".join(steps)
        else:
            # Single modification of a complex core
            return "🔹 **Scaffold Decoration / Derivatization:**\n" + steps[0]
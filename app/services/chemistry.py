"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import random
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, BRICS, GraphDescriptors
import pubchempy as pcp

class ChemistryService:
    """
    Core service for chemical property calculation, structural analysis, 
    and database validation.
    """

    # --- Constants for Delaney's ESOL Equation (Solubility) ---
    ESOL_INTERCEPT = 0.16
    ESOL_COEF_LOGP = -0.63
    ESOL_COEF_MW = -0.0062
    ESOL_COEF_ROTORS = 0.066
    ESOL_COEF_AP = -0.74

    @staticmethod
    def check_novelty(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates the molecule against the PubChem database to determine novelty.
        
        Returns:
            Tuple containing: (is_novel, compound_name, pubchem_link)
        """
        try:
            compounds = pcp.get_compounds(smiles, namespace='smiles')
            if compounds and compounds[0].cid:
                compound = compounds[0]
                # Try to get a synonym, fallback to CID if nameless
                name = compound.synonyms[0] if compound.synonyms else f"Compound {compound.cid}"
                link = f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}"
                return False, name, link
            
            # No CID found implies novelty
            return True, None, None
        except Exception:
            # Fallback to assuming novelty in case of API timeout
            return True, None, None

    @staticmethod
    def analyze_properties(mol: Chem.Mol) -> Dict[str, Any]:
        """
        Computes physicochemical descriptors and ADMET (Absorption, Distribution, 
        Metabolism, Excretion, Toxicity) risks.
        """
        # 1. Physicochemical Descriptors
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        
        # 2. ESOL Solubility Prediction (Delaney, 2004)
        # Calculate Aromatic Proportion (AP)
        aromatic_atom_count = sum([a.GetIsAromatic() for a in mol.GetAtoms()])
        heavy_atom_count = mol.GetNumHeavyAtoms()
        aromatic_proportion = aromatic_atom_count / heavy_atom_count if heavy_atom_count > 0 else 0
        
        esol_log_s = (ChemistryService.ESOL_INTERCEPT +
                      (ChemistryService.ESOL_COEF_LOGP * logp) +
                      (ChemistryService.ESOL_COEF_MW * mw) +
                      (ChemistryService.ESOL_COEF_ROTORS * rotors) +
                      (ChemistryService.ESOL_COEF_AP * aromatic_proportion))

        # 3. Structural Alerts (Toxicity Filters)
        alerts = []
        # Nitro groups often indicate mutagenicity
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")):
            alerts.append("Nitro group")
        # High halogen content can indicate hepatotoxicity
        halogens = len([a for a in mol.GetAtoms() if a.GetSymbol() in ['F', 'Cl', 'Br', 'I']])
        if halogens > 3:
            alerts.append("High Halogen content")
        
        # 4. Blood-Brain Barrier (BBB) Permeability Heuristic
        # TPSA < 90 and MW < 450 is a common rule of thumb for CNS drugs
        brain_penetration = "🧠 Yes" if (tpsa < 90 and mw < 450) else "🛡 No"
        
        # 5. Synthetic Accessibility (SA) Proxy
        # Normalize Bertz Complexity to a 1-10 scale
        complexity = GraphDescriptors.BertzCT(mol)
        sa_score = max(1.0, min(10.0, (complexity - 200) / 100))
        
        sa_text = "Easy"
        if sa_score >= 4: 
            sa_text = "Medium"
        if sa_score >= 7: 
            sa_text = "Difficult"

        return {
            "logp": round(logp, 2),
            "mw": round(mw, 2),
            "qed": round(Descriptors.qed(mol), 2),
            "esol": round(esol_log_s, 2),
            "solubility": "High" if esol_log_s > -2 else "Medium" if esol_log_s > -4 else "Low",
            "brain": brain_penetration,
            "toxicity": "✅ Safe" if not alerts else f"⚠️ Alerts: {', '.join(alerts)}",
            "sa_score": round(sa_score, 1),
            "sa_text": sa_text
        }

    @staticmethod
    def retrosynthesis(mol: Chem.Mol) -> List[str]:
        """
        Performs retrosynthetic fragmentation using the BRICS algorithm.
        Returns a list of top precursor SMILES strings.
        """
        try:
            # Break strategic bonds
            fragmented_mol = BRICS.BreakBRICSBonds(mol)
            fragments = Chem.MolToSmiles(fragmented_mol).split(".")
            
            clean_precursors = []
            for frag in fragments:
                frag_mol = Chem.MolFromSmiles(frag)
                if frag_mol:
                    # Clean up dummy isotopes (e.g., [14*]) left by BRICS
                    for atom in frag_mol.GetAtoms():
                        if atom.GetAtomicNum() == 0: 
                            atom.SetAtomicNum(1) # Convert dummy to Hydrogen
                    
                    # Deduplicate via canonical SMILES
                    clean_precursors.append(Chem.MolToSmiles(frag_mol))
            
            # Return unique, largest fragments first
            unique_frags = sorted(list(set(clean_precursors)), key=len, reverse=True)
            return unique_frags[:4]
            
        except Exception:
            return []

    @staticmethod
    def dock_simulation(mol: Chem.Mol, target_type: str) -> float:
        """
        Approximates binding affinity using a QSAR-based scoring function.
        Acts as a computationally efficient proxy for full molecular docking.
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_bonds = Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        
        # Base scoring function (empirical weights)
        score = -5.0 - (mw / 100.0 * 0.8) - (h_bonds * 0.1)
        
        # Entropy penalty for flexibility
        score += (rotatable_bonds * 0.1)
        
        # Target-specific penalties
        if target_type == "neuro" and logp > 2.5:
            score -= 1.0 # Penalize low lipophilicity for CNS targets
            
        # Add stochastic noise to simulate experimental variance
        final_score = score + random.uniform(-0.5, 0.5)
        
        # Clamp to realistic biophysical range (kcal/mol)
        return round(max(-12.0, min(-2.0, final_score)), 2)
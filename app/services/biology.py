import random
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

class BiologyService:
    """
    Service for estimating bio-physical properties and ligand-protein interactions
    using QSAR (Quantitative Structure-Activity Relationship) heuristics.
    """

    # --- QSAR Coefficients ---
    BASE_AFFINITY = -5.0       # Baseline binding energy (kcal/mol)
    MW_WEIGHT = 0.008          # Contribution of Molecular Weight (Van der Waals)
    HB_WEIGHT = 0.1            # Contribution of Hydrogen Bonds (Enthalpic gain)
    ROTATABLE_PENALTY = 0.1    # Entropic penalty for flexibility
    
    # Target-specific adjustments
    NEURO_LOGP_THRESHOLD = 2.5
    VIRAL_HB_THRESHOLD = 6

    @staticmethod
    def estimate_binding_affinity(mol: Chem.Mol, target_category: str) -> float:
        """
        Approximates the binding affinity (Gibbs free energy, ΔG) based on 
        physicochemical descriptors.
        
        This acts as a high-throughput virtual screening proxy for computationally 
        expensive docking simulations (e.g., AutoDock Vina).

        Args:
            mol: RDKit molecule object.
            target_category: Type of biological target ('neuro', 'viral', etc.).

        Returns:
            float: Estimated binding affinity in kcal/mol (lower is better).
        """
        
        # 1. Calculate Physicochemical Descriptors
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        # Sum of donors and acceptors represents polar interaction potential
        h_bonds = Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)
        rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        
        # 2. Base Score Calculation
        # Heuristic: Heavier molecules often have better VdW contacts, 
        # but high flexibility (rotors) incurs an entropic penalty upon binding.
        score = (BiologyService.BASE_AFFINITY 
                 - (mw * BiologyService.MW_WEIGHT) 
                 - (h_bonds * BiologyService.HB_WEIGHT) 
                 + (rotatable_bonds * BiologyService.ROTATABLE_PENALTY))
        
        # 3. Target-Specific Optimization
        # Adjust score based on known pharmacophore preferences
        if target_category == "neuro":
            # CNS drugs require higher lipophilicity to cross the Blood-Brain Barrier (BBB)
            if logp > BiologyService.NEURO_LOGP_THRESHOLD: 
                score -= 1.0 
        
        elif target_category == "viral":
            # Viral proteases often possess polar active sites requiring H-bond networks
            if h_bonds > BiologyService.VIRAL_HB_THRESHOLD:
                score -= 1.0
            
        # 4. Stochastic Variance
        # Adds noise to simulate the uncertainty inherent in QSAR approximations
        noise = random.uniform(-0.5, 0.5)
        final_score = score + noise
        
        # 5. Physical Constraints
        # Clamp values to realistic ranges for small molecule drugs (-3.0 to -12.0 kcal/mol)
        return round(max(-12.0, min(-3.0, final_score)), 2)

    @staticmethod
    def interpret_affinity(score: float) -> str:
        """
        Classifies the calculated affinity into human-readable categories.
        """
        if score > -6.0:
            return "❌ Weak Binding (> -6.0)"
        if score > -7.5:
            return "⚠️ Moderate Affinity"
        if score > -9.0:
            return "✅ Strong Binding (Lead Candidate)"
        
        return "💎 **Potent Inhibitor (Nanomolar Range)**"
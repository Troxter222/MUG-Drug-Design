"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.MolStandardize import rdMolStandardize
import pubchempy as pcp

class ChemistryService:
    
    # --- SMARTS для pKa эвристики ---
    ACIDIC_GRP = Chem.MolFromSmarts("[CX3](=O)[OH]")
    BASIC_AMINE = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]") 
    BASIC_AROM = Chem.MolFromSmarts("[n]")

    @staticmethod
    def normalize_structure(mol: Chem.Mol) -> Chem.Mol:
        try:
            clean = rdMolStandardize.Cleanup(mol)
            enumerator = rdMolStandardize.TautomerEnumerator()
            canon = enumerator.Canonicalize(clean)
            return rdMolStandardize.ChargeParent(canon)
        except Exception: 
            return mol

    @staticmethod
    def estimate_pka_logd(mol: Chem.Mol, logp: float) -> Tuple[float, str, float]:
        """
        Approximates LogD at pH 7.4 using Henderson-Hasselbalch equation
        based on identified functional groups.
        
        Returns: (cLogD_7.4, pKa_description, most_basic_pka)
        """
        # Эмпирические значения pKa
        pka_acid = 4.5  # Типичная карбоновая кислота
        pka_base = 9.5  # Типичный алифатический амин
        
        has_acid = mol.HasSubstructMatch(ChemistryService.ACIDIC_GRP)
        has_base = mol.HasSubstructMatch(ChemistryService.BASIC_AMINE)
        
        log_d = logp
        pka_desc = "Neutral"
        most_basic = 0.0

        # Корректировка LogD (учитываем ионизацию)
        # Если есть кислота: при pH 7.4 она заряжена (-) -> LogD падает
        if has_acid:
            # Fraction ionized for acid: 1 / (1 + 10^(pKa - pH))
            # LogD = LogP - log10(1 + 10^(pH - pKa))
            correction = math.log10(1 + 10**(7.4 - pka_acid))
            log_d -= correction
            pka_desc = f"Acidic (pKa~{pka_acid})"

        # Если есть база: при pH 7.4 она заряжена (+) -> LogD падает
        if has_base:
            # Fraction ionized for base: 1 / (1 + 10^(pH - pKa))
            correction = math.log10(1 + 10**(pka_base - 7.4))
            log_d -= correction
            if has_acid:
                pka_desc = "Zwitterionic (Acid+Base)"
            else:
                pka_desc = f"Basic (pKa~{pka_base})"
            most_basic = pka_base

        return round(log_d, 2), pka_desc, most_basic

    @staticmethod
    def validate_cns_rules(mol: Chem.Mol, target_cat: str) -> Tuple[float, List[str]]:
        penalty = 0.0
        warnings = []
        if "neuro" not in target_cat: 
            return 0.0, []

        # Zwitterion check
        if mol.HasSubstructMatch(ChemistryService.ACIDIC_GRP) and \
           mol.HasSubstructMatch(ChemistryService.BASIC_AMINE):
            penalty += 40.0
            warnings.append("Zwitterion (Likely BBB Impermeable)")

        return penalty, warnings

    @staticmethod
    def check_novelty(smiles: str) -> Tuple[bool, Optional[str], Optional[str]]:
        try:
            compounds = pcp.get_compounds(smiles, namespace='smiles')
            if compounds and compounds[0].cid:
                c = compounds[0]
                name = c.synonyms[0] if c.synonyms else f"Compound {c.cid}"
                link = f"https://pubchem.ncbi.nlm.nih.gov/compound/{c.cid}"
                return False, name, link
            return True, None, None
        except Exception:
            return True, None, None

    @staticmethod
    def analyze_properties(mol: Chem.Mol) -> Dict[str, Any]:
        """
        Enhanced analysis with Consistency Rules.
        """
        # 1. Descriptors
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        qed = Descriptors.qed(mol)
        
        # 2. Advanced: cLogD 7.4 Calculation
        clogd, pka_type, basic_pka = ChemistryService.estimate_pka_logd(mol, logp)

        # 3. Alerts
        alerts = []
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")): 
            alerts.append("Nitro")
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[I,Br,Cl]")): 
            alerts.append("Halogen")
        
        # SOFTENED TERMINOLOGY
        risk_profile = "No obvious alerts (Bertz)" if not alerts else f"Alerts: {','.join(alerts)}"

        # 4. CNS CONSISTENCY LOGIC (Gating)
        # Правило: Если LogD < 0 (слишком полярно/ионизировано), пассивный транспорт невозможен
        
        cns_score = 0
        cns_reason = []

        # (A) LogD Check (Главный Гейткипер)
        if clogd < 0:
            cns_reason.append("Too Hydrophilic/Ionized")
            cns_score -= 5 # Force Low
        elif clogd > 5:
            cns_reason.append("Too Lipophilic")
            cns_score -= 2
        else:
            cns_score += 2 # Good range (0-5)

        # (B) MW Check
        if mw < 450: 
            cns_score += 1
        else: 
            cns_reason.append("High MW")

        # (C) TPSA Check
        if tpsa < 90:
            cns_score += 1
        elif tpsa > 120: 
            cns_score -= 2
            cns_reason.append("High TPSA")

        # Final Verdict
        if cns_score >= 4:
            cns_prob = "High (Ideal)"
        elif cns_score >= 1:
            cns_prob = "Medium"
        else:
            reason_str = f" ({', '.join(cns_reason)})" if cns_reason else ""
            cns_prob = f"Low{reason_str}"

        # 5. Ionization Note (State)
        state_desc = pka_type
        if "Basic" in pka_type and clogd < 0:
             state_desc += " (Protonated @ pH 7.4)"

        return {
            "mw": round(mw, 1),
            "logp": round(logp, 2),
            "clogd": clogd,
            "pka_type": state_desc,
            "tpsa": round(tpsa, 1),
            "qed": round(qed, 2),
            "cns_prob": cns_prob,
            "risk_profile": risk_profile,
            "alerts_list": alerts,
            "complexity_ct": round(Descriptors.BertzCT(mol), 1)
        }
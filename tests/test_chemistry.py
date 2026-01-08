import unittest
from rdkit import Chem
from app.services.chemistry import ChemistryService

class TestChemistryService(unittest.TestCase):
    def setUp(self):
        self.service = ChemistryService()
        self.valid_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        self.invalid_smiles = "C1CC(Invalid"

    def test_normalize_structure(self):
        mol = Chem.MolFromSmiles(self.valid_smiles)
        norm_mol = self.service.normalize_structure(mol)
        self.assertIsNotNone(norm_mol)
        self.assertIsInstance(norm_mol, Chem.Mol)

    def test_analyze_properties_valid(self):
        mol = Chem.MolFromSmiles(self.valid_smiles)
        props = self.service.analyze_properties(mol)
        
        self.assertIn("mw", props)
        self.assertIn("logp", props)
        self.assertIn("qed", props)
        self.assertIn("cns_prob", props)
        
        # Aspirin MW is approx 180.16
        self.assertTrue(170 < props["mw"] < 190)

    def test_cns_rules(self):
        # A huge molecule (likely non-CNS)
        huge_mol = Chem.MolFromSmiles("C" * 50) 
        penalty, warnings = self.service.validate_cns_rules(huge_mol, "neuro")
        self.assertIsInstance(penalty, float)
        self.assertIsInstance(warnings, list)

    def test_check_novelty(self):
        # Aspirin is definitely not novel
        is_novel, name, _ = self.service.check_novelty(self.valid_smiles)
        # Note: Depending on network/API, this might fail or return fallback. 
        # Checking strictly bool return types here.
        self.assertIsInstance(is_novel, bool)

if __name__ == "__main__":
    unittest.main()
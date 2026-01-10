import unittest
from rdkit import Chem
from app.services.retrosynthesis import RetrosynthesisService

class TestRetrosynthesis(unittest.TestCase):
    def test_fragmentation_simple(self):
        mol = Chem.MolFromSmiles("CCO")
        frags = RetrosynthesisService.get_building_blocks(mol)
        self.assertIsInstance(frags, list)

    def test_fragmentation_complex(self):
        mol = Chem.MolFromSmiles("O=C(Nc1ccccc1)c1ccccc1") 
        frags = RetrosynthesisService.get_building_blocks(mol)
        self.assertTrue(len(frags) >= 1)
        for f in frags:
            self.assertIsNotNone(Chem.MolFromSmiles(f))

    def test_complexity_assessment(self):
        simple_mol = Chem.MolFromSmiles("C")
        complex_mol = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45")
        
        score_simple = RetrosynthesisService.assess_complexity(simple_mol)
        score_complex = RetrosynthesisService.assess_complexity(complex_mol)
        
        self.assertEqual(score_simple, "Low")
        self.assertIn(score_complex, ["Medium", "High (Fused/Complex System)"])

    def test_synthesis_description(self):
        # Case 1: 2 fragments (Simple)
        desc_simple = RetrosynthesisService.describe_synthesis(["frag1", "frag2"], "Low")
        self.assertIn("Strategy", desc_simple)
        self.assertIn("Simple Derivatization", desc_simple) 
        self.assertIn("High", desc_simple)

        # Case 2: 4 fragments (Convergent)
        desc_conv = RetrosynthesisService.describe_synthesis(["f1", "f2", "f3", "f4"], "Low")
        self.assertIn("Convergent", desc_conv)

if __name__ == "__main__":
    unittest.main()
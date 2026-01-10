import unittest
from rdkit import Chem
from unittest.mock import MagicMock, patch
from app.services.biology import BiologyService

class TestBiologyService(unittest.TestCase):
    def setUp(self):
        self.service = BiologyService()
        # Force QSAR fallback mode for testing logic regardless of Vina installation
        self.service.vina_ready = False 
        self.mol_small = Chem.MolFromSmiles("CC") # Ethane (Small)
        self.mol_large = Chem.MolFromSmiles("C1CCCCC1C2CCCCC2C3CCCCC3") # Large hydrophobic

    def test_qsar_fallback_logic(self):
        # Fallback formula depends on MW. Heavier molecules should generally bind stronger (more negative)
        score_small = self.service._qsar_fallback(self.mol_small)
        score_large = self.service._qsar_fallback(self.mol_large)
        
        self.assertIsInstance(score_small, float)
        self.assertIsInstance(score_large, float)
        
        # Ensure scores are within reasonable docking range (-15 to -3)
        self.assertTrue(-15.0 <= score_small <= -3.0)
        
        # Larger molecule implies more interactions -> lower score in this simple heuristic
        # Note: The formula adds random noise, so we use a loose comparison or mock random
        with patch('random.uniform', return_value=0.0):
            clean_score_small = self.service._qsar_fallback(self.mol_small)
            clean_score_large = self.service._qsar_fallback(self.mol_large)
            self.assertLess(clean_score_large, clean_score_small)

    def test_interpret_affinity(self):
        self.assertEqual(BiologyService.interpret_affinity(-11.0), "Potent Inhibitor (Optimized)")
        self.assertEqual(BiologyService.interpret_affinity(-5.0), "Non-binder / Very Weak")

    def test_confidence_level(self):
        # High affinity -> High confidence
        self.assertEqual(BiologyService.get_confidence_level(-10.0, 0.8), "High (3D Validated)")
        # Low affinity -> Low confidence
        self.assertEqual(BiologyService.get_confidence_level(-5.0, 0.2), "Low")

if __name__ == "__main__":
    unittest.main()
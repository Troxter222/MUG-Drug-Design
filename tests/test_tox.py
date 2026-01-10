import unittest
import numpy as np
from rdkit import Chem
from unittest.mock import MagicMock
from app.services.toxicity import ToxicityService

class TestToxicityService(unittest.TestCase):
    def setUp(self):
        self.service = ToxicityService()
        self.mol = Chem.MolFromSmiles("c1ccccc1O") 

    def test_prediction_logic_safe(self):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        
        self.service.models = {'SR-p53': mock_model}
        self.service.loaded = True
        
        risks = self.service.predict(self.mol)
        self.assertEqual(len(risks), 0, "Should have no risks for safe prediction")

    def test_prediction_logic_toxic(self):
        mock_model = MagicMock()
        # Mocking high probability for class 1 (Toxic)
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        
        self.service.models = {'NR-AhR': mock_model}
        self.service.loaded = True
        
        risks = self.service.predict(self.mol)
        
        self.assertEqual(len(risks), 1)
        # Check for the human-readable label part produced by the service
        self.assertIn("Toxin Response", risks[0]) 
        self.assertIn("AhR", risks[0])
        self.assertIn("High", risks[0])
        self.assertIn("🔴", risks[0])

    def test_graceful_fail_on_invalid_mol(self):
        risks = self.service.predict(None)
        self.assertEqual(risks, [])

if __name__ == "__main__":
    unittest.main()
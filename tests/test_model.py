import unittest
import torch
from app.core.transformer_model import MoleculeTransformer

class TestMoleculeTransformer(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 20
        self.d_model = 32
        self.latent = 16
        self.device = torch.device("cpu")
        
        self.model = MoleculeTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=64,
            latent_size=self.latent
        ).to(self.device)

    def test_forward_pass_shapes(self):
        # Batch size 2, Seq len 10
        src = torch.randint(0, self.vocab_size, (10, 2)).to(self.device)
        tgt = torch.randint(0, self.vocab_size, (10, 2)).to(self.device)
        
        logits, mu, logvar = self.model(src, tgt)
        
        # Logits should be [seq_len, batch, vocab_size]
        self.assertEqual(logits.shape, (10, 2, self.vocab_size))
        # Latent params should be [batch, latent_size]
        self.assertEqual(mu.shape, (2, self.latent))
        self.assertEqual(logvar.shape, (2, self.latent))

    def test_sampling_shapes(self):
        # Mock vocab object needed for sampling
        class MockVocab:
            sos_idx = 1
            eos_idx = 2
        
        n_samples = 3
        max_len = 15
        
        samples = self.model.sample(
            n_samples=n_samples,
            device=self.device,
            vocab=MockVocab(),
            max_len=max_len
        )
        
        # Expected shape [batch, seq_len] (transposed internally in sample method)
        # Note: Check implementation of sample return. Usually it returns [batch, seq]
        self.assertEqual(samples.shape[0], n_samples)
        self.assertTrue(samples.shape[1] <= max_len)

if __name__ == "__main__":
    unittest.main()
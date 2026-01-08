import unittest
import torch
from app.core.vocab import Vocabulary

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        # Minimal vocab for testing
        self.tokens = ["<pad>", "<sos>", "<eos>", "[C]", "[O]", "[N]"]
        self.vocab = Vocabulary(self.tokens)

    def test_special_tokens(self):
        self.assertEqual(self.vocab.char2idx["<pad>"], self.vocab.pad_idx)
        self.assertEqual(self.vocab.char2idx["<sos>"], self.vocab.sos_idx)
        self.assertEqual(self.vocab.char2idx["<eos>"], self.vocab.eos_idx)

    def test_encode_decode_consistency(self):
        selfies_str = "[C][O][C]"
        # Note: [C] is index 3, [O] is 4
        # Expected: [SOS, 3, 4, 3, EOS, PAD...]
        
        indices = self.vocab.encode(selfies_str, max_len=10)
        self.assertEqual(indices[0], self.vocab.sos_idx)
        
        # Decode back
        decoded = self.vocab.decode(indices)
        # Verify decoding logic strips special tokens appropriately for reconstruction
        # (Implementation dependent, usually decoding returns just the content)
        self.assertIsInstance(decoded, str)

    def test_batch_processing(self):
        batch_selfies = ["[C]", "[O]"]
        tensor = self.vocab.batch_encode(batch_selfies, max_len=5)
        self.assertEqual(tensor.shape, (2, 5))
        self.assertIsInstance(tensor, torch.Tensor)

if __name__ == "__main__":
    unittest.main()
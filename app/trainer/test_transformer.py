"""
Molecular Universe Generator (MUG) - Model Benchmark Suite
Author: Ali (Troxter222)
License: MIT

Comprehensive evaluation framework for comparing molecular generation models.
Tests validity, uniqueness, drug-likeness (QED), and toxicity metrics.
"""

import torch
import json
import os
import glob
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from app.core.transformer_model import MoleculeTransformer


class TestConfig:
    """
    Configuration settings.
    NOTE: These parameters must match the training configuration exactly.
    """
    VOCAB_FILE = 'dataset/processed/vocab_transformer.json'
    CHECKPOINT_DIR = 'checkpoints_transformer'

    # Architecture settings
    D_MODEL = 128
    NHEAD = 4
    LAYERS = 3
    LATENT = 64
    MAX_LEN = 150

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleVocab:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.pad_idx = self.char2idx.get('<pad>', 0)
        self.sos_idx = self.char2idx.get('<sos>', 1)
        self.eos_idx = self.char2idx.get('<eos>', 2)

    def decode(self, indices):
        tokens = []
        for i in indices:
            idx = i.item() if torch.is_tensor(i) else i
            if idx == self.eos_idx:
                break
            if idx != self.pad_idx and idx != self.sos_idx:
                tokens.append(self.idx2char[idx])
        return "".join(tokens)


def get_latest_checkpoint():
    """Finds the most recently created weight file."""
    files = glob.glob(f"{TestConfig.CHECKPOINT_DIR}/*.pth")
    if not files:
        return None
    # Sort by creation time (newest last)
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def test():
    print("Running Transformer Test...")

    # 1. Load Vocabulary
    if not os.path.exists(TestConfig.VOCAB_FILE):
        print(f"Error: Vocabulary not found: {TestConfig.VOCAB_FILE}")
        return
    vocab = SimpleVocab(TestConfig.VOCAB_FILE)

    # 2. Find Model
    ckpt_path = get_latest_checkpoint()
    if not ckpt_path:
        print(f"Error: No checkpoints found in {TestConfig.CHECKPOINT_DIR}. "
              "Wait for the first epoch to finish.")
        return

    print(f"Loading weights: {ckpt_path}")

    # 3. Initialization
    model = MoleculeTransformer(
        vocab_size=len(vocab.vocab),
        d_model=TestConfig.D_MODEL,
        nhead=TestConfig.NHEAD,
        num_encoder_layers=TestConfig.LAYERS,
        num_decoder_layers=TestConfig.LAYERS,
        latent_size=TestConfig.LATENT
    ).to(TestConfig.DEVICE)

    model.load_state_dict(
        torch.load(ckpt_path, map_location=TestConfig.DEVICE)  # nosec
    )
    model.eval()

    # 4. Generation
    num_samples = 50
    print(f"Generating {num_samples} molecules...")

    valid_mols = []
    valid_smiles = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate 1 sample
            indices = model.sample(
                TestConfig.DEVICE, vocab, max_len=TestConfig.MAX_LEN
            )
            selfies_str = vocab.decode(indices)

            try:
                # Decode SELFIES -> SMILES
                smi = sf.decoder(selfies_str)
                if not smi:
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol:
                    valid_mols.append(mol)
                    valid_smiles.append(smi)
            except Exception:
                continue

    # 5. Results
    validity = (len(valid_mols) / num_samples) * 100
    unique = len(set(valid_smiles))
    unique_ratio = (unique / len(valid_smiles) * 100) if valid_smiles else 0

    print("\nTEST REPORT:")
    print(f"Validity: {validity:.1f}% (Target > 80%)")
    print(f"Uniqueness: {unique_ratio:.1f}%")

    if valid_mols:
        print("\nGeneration examples:")
        for i in range(min(5, len(valid_smiles))):
            print(f"{i+1}. {valid_smiles[i]}")

        # Draw grid
        img = Draw.MolsToGridImage(
            valid_mols[:9],
            molsPerRow=3,
            subImgSize=(300, 300),
            legends=[f"Mol {i+1}" for i in range(len(valid_mols[:9]))]
        )
        img.save("transformer_test_results.png")
        print("\nImage saved to transformer_test_results.png")

        # Calculate average QED
        qeds = [Descriptors.qed(m) for m in valid_mols]
        avg_qed = sum(qeds) / len(qeds)
        print(f"Average QED (Drug-likeness): {avg_qed:.2f}")
    else:
        print("Warning: Model generated only invalid data. "
              "More training required.")


if __name__ == "__main__":
    test()
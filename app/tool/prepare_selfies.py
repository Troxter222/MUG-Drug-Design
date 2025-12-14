import json
import pandas as pd
import selfies as sf
from tqdm import tqdm

# Paths
INPUT_FILE = "data/processed/train.csv"
output_file = "data/processed/train_selfies.csv"
VOCAB_FILE = "data/processed/vocab_selfies.json"


def preprocess_selfies():
    print("Starting SMILES → SELFIES conversion...")

    df = pd.read_csv(INPUT_FILE)
    smiles_list = df["SMILES"].tolist()

    selfies_list = []
    vocab_set = set()

    print("Encoding SMILES into SELFIES...")
    for smi in tqdm(smiles_list):
        try:
            selfie = sf.encoder(smi)

            if selfie:
                selfies_list.append(selfie)

                tokens = list(sf.split_selfies(selfie))
                vocab_set.update(tokens)

        except Exception:
            continue

    # Save converted dataset
    new_df = pd.DataFrame({"SELFIES": selfies_list})
    new_df.to_csv(output_file, index=False)

    # Build vocabulary with special tokens
    special_tokens = ["<pad>", "<sos>", "<eos>"]
    final_vocab = special_tokens + sorted(list(vocab_set))

    with open(VOCAB_FILE, "w") as f:
        json.dump(final_vocab, f, indent=2)

    print("Processing completed.")
    print(f"Total molecules converted: {len(selfies_list)}")
    print(f"Vocabulary size: {len(final_vocab)} tokens")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    preprocess_selfies()

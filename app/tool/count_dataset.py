"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import os
import pandas as pd
import glob

# Paths
DATA_DIR = "data/gnn_datasets"


def analyze_dataset():
    print("Running dataset inspection...\n")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        print("Run download_real_data.py first.")
        return

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if not csv_files:
        print("No CSV files found in the dataset directory.")
        return

    total_samples = 0

    print(f"{'FILE':<30} | {'SAMPLES':<15} | {'COLUMNS'}")
    print("-" * 100)

    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path)
            count = len(df)

            preview_cols = ", ".join(df.columns[:3])
            if len(df.columns) > 3:
                preview_cols += "..."

            print(f"{file_name:<30} | {count:<15} | {preview_cols}")
            total_samples += count

        except Exception as e:
            print(f"{file_name:<30} | ERROR          | {e}")

    print("-" * 100)
    print(f"\nTotal samples collected: {total_samples}")

    if total_samples > 0:
        print("Dataset is ready for GNN training.")
    else:
        print("Dataset is empty. Run download_real_data.py.")


if __name__ == "__main__":
    analyze_dataset()

import os
import glob
import csv
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# ===== Настройки =====
INPUT_PATTERN = "tmp_*.txt"
OUT_FILE = "dataset/processed/transformer_train_v2.csv"

# Сделай максимально мягкие фильтры, чтобы не терять данные
MIN_LEN = 1
MAX_LEN = 1000

CANONICALIZE = True

# сколько строк взять для авто-определения колонки
SNIFF_LINES = 5000

RDLogger.DisableLog("rdApp.*")  # чтобы не спамило "SMILES Parse Error"


def is_valid_smiles(s: str) -> bool:
    if not s:
        return False
    mol = Chem.MolFromSmiles(s)
    return mol is not None


def sniff_smiles_column(file_path: str, sniff_lines: int = 5000) -> int:
    """
    Определяет индекс колонки (0,1,2,...) где чаще всего встречается валидный SMILES.
    """
    valid_counts = {}   # col_idx -> valid
    total_counts = {}   # col_idx -> seen

    checked = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if checked >= sniff_lines:
                break

            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("smiles"):
                continue

            parts = s.split()
            if len(parts) == 0:
                continue

            # пробуем все колонки, которые есть в строке
            for i, tok in enumerate(parts[:10]):  # ограничим до первых 10 колонок
                total_counts[i] = total_counts.get(i, 0) + 1
                if is_valid_smiles(tok):
                    valid_counts[i] = valid_counts.get(i, 0) + 1

            checked += 1

    if not total_counts:
        return 0

    # выбираем колонку с максимальной долей валидных
    best_i = 0
    best_ratio = -1.0
    for i in total_counts:
        v = valid_counts.get(i, 0)
        t = total_counts[i]
        ratio = v / t if t else 0.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_i = i

    print("🔎 Auto-detect SMILES column:")
    for i in sorted(total_counts):
        v = valid_counts.get(i, 0)
        t = total_counts[i]
        print(f"  col[{i}]: valid {v}/{t} = {100*v/t:.2f}%")

    print(f"✅ Selected column index: {best_i}")
    return best_i


def main():
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        print("❌ Файлы tmp_*.txt не найдены!")
        return

    print(f"📂 Найдено файлов: {len(files)}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    total_lines = 0
    parsed_lines = 0
    too_short = 0
    too_long = 0
    rdkit_invalid = 0
    rdkit_valid = 0

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["smiles"])

        for fp in files:
            print(f"\n📄 File: {fp}")
            smiles_col = sniff_smiles_column(fp, SNIFF_LINES)

            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in tqdm(f, desc=os.path.basename(fp), unit=" lines"):
                    total_lines += 1

                    s = line.strip()
                    if not s:
                        continue
                    if s.lower().startswith("smiles"):
                        continue

                    parts = s.split()
                    if len(parts) <= smiles_col:
                        continue

                    smi = parts[smiles_col]
                    parsed_lines += 1

                    if len(smi) < MIN_LEN:
                        too_short += 1
                        continue
                    if len(smi) > MAX_LEN:
                        too_long += 1
                        continue

                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        rdkit_invalid += 1
                        continue

                    rdkit_valid += 1
                    if CANONICALIZE:
                        smi = Chem.MolToSmiles(mol, canonical=True)

                    writer.writerow([smi])

    print("\n===== REPORT =====")
    print(f"Total raw lines:        {total_lines}")
    print(f"Parsed (had token):     {parsed_lines}")
    print(f"Dropped too short:      {too_short}")
    print(f"Dropped too long:       {too_long}")
    print(f"RDKit invalid:          {rdkit_invalid}")
    print(f"RDKit valid (written):  {rdkit_valid}")
    print(f"\n🎉 CSV saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()

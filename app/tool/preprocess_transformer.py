"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import selfies as sf
from tqdm import tqdm

# --- КОНФИГУРАЦИЯ ---
class PreprocessConfig:
    """Configuration for data preprocessing pipeline."""
    
    # ВХОДНОЙ ФАЙЛ (Тот, который создал build_dataset.py)
    # Если у вас файл называется transformer_train_v2.csv, поменяйте название здесь!
    INPUT_FILE = Path("dataset/processed/transformer_train_v2.csv")
    
    # ВЫХОДНАЯ ПАПКА (Куда сохранять готовое)
    OUTPUT_DIR = Path("dataset/processed_v2")
    
    # Имена выходных файлов
    OUTPUT_FILE = OUTPUT_DIR / "transformer_train.csv"
    VOCAB_FILE = OUTPUT_DIR / "vocab_transformer.json"
    
    # Параметры обработки
    MAX_SEQUENCE_LENGTH = 100  # Максимальная длина SELFIES токенов (оптимизация под GTX 1650)
    MIN_SEQUENCE_LENGTH = 5
    SMILES_COLUMN = "smiles"   # Название колонки в входном CSV
    
    # Специальные токены (Обязательно!)
    SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>']
    
    LOG_LEVEL = logging.INFO

def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=PreprocessConfig.LOG_LEVEL,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("MUG.Preprocessing")

def validate_input_file(file_path: Path, logger: logging.Logger) -> bool:
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        return False
    return True

def process_molecules(input_file: Path, logger: logging.Logger) -> Tuple[List[str], Dict[str, int]]:
    """
    Основной цикл конвертации SMILES -> SELFIES.
    """
    logger.info(f"Loading molecular dataset: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return [], {}

    if PreprocessConfig.SMILES_COLUMN not in df.columns:
        # Пытаемся найти колонку, если имя отличается
        cols = df.columns.tolist()
        if len(cols) > 0:
            target_col = cols[0] # Берем первую
        else:
            logger.error("CSV file is empty or has no columns.")
            return [], {}
    else:
        target_col = PreprocessConfig.SMILES_COLUMN

    total_molecules = len(df)
    logger.info(f"Total molecules loaded: {total_molecules:,}")
    
    valid_selfies = []
    stats = {
        'total': total_molecules,
        'converted': 0,
        'failed_conversion': 0,
        'too_long': 0,
        'too_short': 0,
        'invalid': 0
    }
    
    logger.info("Converting SMILES to SELFIES format...")
    
    # Используем tqdm для прогресс-бара
    for smiles in tqdm(df[target_col].astype(str), desc="Processing"):
        try:
            # 1. Конвертация
            selfies = sf.encoder(smiles)
            if not selfies:
                stats['failed_conversion'] += 1
                continue
            
            # 2. Проверка длины (считаем количество токенов)
            # Это важно, чтобы модель не падала по памяти
            tokens = list(sf.split_selfies(selfies))
            length = len(tokens)
            
            if length > PreprocessConfig.MAX_SEQUENCE_LENGTH:
                stats['too_long'] += 1
                continue
            if length < PreprocessConfig.MIN_SEQUENCE_LENGTH:
                stats['too_short'] += 1
                continue
                
            valid_selfies.append(selfies)
            stats['converted'] += 1
            
        except Exception:
            stats['failed_conversion'] += 1
            continue
    
    return valid_selfies, stats

def extract_vocabulary(selfies_list: List[str], logger: logging.Logger) -> Set[str]:
    """Собирает все уникальные токены из списка SELFIES."""
    logger.info("Building vocabulary from molecular sequences...")
    vocab_set = set()
    
    for selfies in tqdm(selfies_list, desc="Extracting tokens"):
        try:
            tokens = sf.split_selfies(selfies)
            vocab_set.update(tokens)
        except Exception:
            continue
    
    return vocab_set

def save_data(selfies_list: List[str], vocab_set: Set[str], logger: logging.Logger):
    """Сохраняет CSV и JSON словаря."""
    
    # 1. Сохраняем CSV
    logger.info(f"Saving processed data to: {PreprocessConfig.OUTPUT_FILE}")
    df = pd.DataFrame({'selfies': selfies_list})
    df.to_csv(PreprocessConfig.OUTPUT_FILE, index=False)
    logger.info(f"Successfully saved {len(selfies_list):,} molecules")
    
    # 2. Сохраняем Словарь
    logger.info("Building vocabulary...")
    # Сортируем и добавляем спец. токены в начало
    final_vocab = PreprocessConfig.SPECIAL_TOKENS + sorted(list(vocab_set))
    
    logger.info(f"Vocabulary size: {len(final_vocab)} tokens")
    logger.info(f"  - Special tokens: {len(PreprocessConfig.SPECIAL_TOKENS)}")
    logger.info(f"  - Molecular tokens: {len(vocab_set)}")
    
    with open(PreprocessConfig.VOCAB_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_vocab, f, indent=2)
        
    logger.info(f"Vocabulary saved to: {PreprocessConfig.VOCAB_FILE}")
    
    # Показываем примеры
    print_samples(selfies_list, final_vocab, logger)

def print_samples(selfies_list, vocab, logger):
    logger.info("\n" + "="*60)
    logger.info("Sample Molecules (SELFIES)")
    logger.info("="*60)
    for i, s in enumerate(selfies_list[:3], 1):
        tokens = list(sf.split_selfies(s))
        logger.info(f"\nMolecule {i}:")
        logger.info(f"  Length: {len(tokens)} tokens")
        logger.info(f"  SELFIES: {s[:100]}...")
        
    logger.info("\n" + "="*60)
    logger.info("Vocabulary Sample (First 20 tokens)")
    logger.info("="*60)
    logger.info(f"{vocab[:20]}")
    logger.info("="*60 + "\n")

def print_statistics(stats: Dict[str, int], logger: logging.Logger):
    logger.info("\n" + "="*60)
    logger.info("Processing Statistics")
    logger.info("="*60)
    logger.info(f"Total molecules:            {stats['total']:>10,}")
    logger.info(f"Successfully converted:     {stats['converted']:>10,} ({stats['converted']/stats['total']*100:.2f}%)")
    logger.info(f"Failed conversion:          {stats['failed_conversion']:>10,}")
    logger.info(f"Sequences too long:         {stats['too_long']:>10,}")
    logger.info(f"Sequences too short:        {stats['too_short']:>10,}")
    logger.info("="*60 + "\n")

def main():
    logger = setup_logger()
    logger.info("="*60)
    logger.info("MUG Data Preprocessing Pipeline")
    logger.info("="*60)
    
    # Проверка путей
    if not validate_input_file(PreprocessConfig.INPUT_FILE, logger):
        return
    
    PreprocessConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {PreprocessConfig.OUTPUT_DIR}")
    
    # Обработка
    selfies_list, stats = process_molecules(PreprocessConfig.INPUT_FILE, logger)
    
    if not selfies_list:
        logger.error("No valid molecules processed. Check input file format.")
        return

    print_statistics(stats, logger)
    
    # Словарь
    vocab_set = extract_vocabulary(selfies_list, logger)
    
    # Сохранение
    save_data(selfies_list, vocab_set, logger)
    
    logger.info("✅ Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
"""
Molecular Universe Generator (MUG) - Data Preprocessing Pipeline
Author: Ali (Troxter222)
License: MIT

Converts SMILES chemical notation to SELFIES representation and builds vocabulary
for transformer-based molecular generation models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import selfies as sf
from tqdm import tqdm


# Configuration
class PreprocessConfig:
    """Configuration for data preprocessing pipeline."""
    
    INPUT_FILE = Path("dataset/raw/pretrain/global_chem_space.csv")
    OUTPUT_DIR = Path("dataset/processed")
    OUTPUT_FILE = OUTPUT_DIR / "transformer_train.csv"
    VOCAB_FILE = OUTPUT_DIR / "vocab_transformer.json"
    
    # Processing parameters
    MAX_SEQUENCE_LENGTH = 150
    MIN_SEQUENCE_LENGTH = 5
    SMILES_COLUMN = "smiles"
    
    # Special tokens
    SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>']
    
    # Logging
    LOG_LEVEL = logging.INFO


def setup_logger() -> logging.Logger:
    """Configure logging for preprocessing pipeline."""
    logging.basicConfig(
        level=PreprocessConfig.LOG_LEVEL,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("MUG.Preprocessing")


def validate_input_file(file_path: Path, logger: logging.Logger) -> bool:
    """
    Validate input CSV file exists and has required columns.
    
    Args:
        file_path: Path to input CSV file
        logger: Logger instance
        
    Returns:
        True if validation passes, False otherwise
    """
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        logger.info("Please place your molecular dataset at the specified location.")
        return False
    
    try:
        df = pd.read_csv(file_path, nrows=5)
        if PreprocessConfig.SMILES_COLUMN not in df.columns:
            logger.error(
                f"Required column '{PreprocessConfig.SMILES_COLUMN}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )
            return False
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return False
    
    return True


def smiles_to_selfies(smiles: str) -> Optional[str]:
    """
    Convert SMILES string to SELFIES representation.
    
    Args:
        smiles: SMILES molecular notation
        
    Returns:
        SELFIES string or None if conversion fails
    """
    try:
        selfies = sf.encoder(smiles)
        return selfies if selfies else None
    except Exception:
        return None


def validate_selfies(
    selfies: str,
    max_len: int,
    min_len: int
) -> bool:
    """
    Validate SELFIES string meets length requirements.
    
    Args:
        selfies: SELFIES representation
        max_len: Maximum allowed sequence length
        min_len: Minimum allowed sequence length
        
    Returns:
        True if valid, False otherwise
    """
    try:
        tokens = list(sf.split_selfies(selfies))
        length = len(tokens)
        return min_len <= length <= max_len
    except Exception:
        return False


def extract_vocabulary(selfies_list: List[str], logger: logging.Logger) -> Set[str]:
    """
    Extract unique tokens from SELFIES sequences.
    
    Args:
        selfies_list: List of SELFIES strings
        logger: Logger instance
        
    Returns:
        Set of unique tokens
    """
    logger.info("Building vocabulary from molecular sequences...")
    vocab_set = set()
    
    for selfies in tqdm(selfies_list, desc="Extracting tokens"):
        try:
            tokens = sf.split_selfies(selfies)
            vocab_set.update(tokens)
        except Exception:
            continue
    
    return vocab_set


def process_molecules(
    input_file: Path,
    logger: logging.Logger
) -> Tuple[List[str], Dict[str, int]]:
    """
    Convert SMILES molecules to SELFIES format with validation.
    
    Args:
        input_file: Path to input CSV file
        logger: Logger instance
        
    Returns:
        Tuple of (valid_selfies_list, conversion_statistics)
    """
    logger.info(f"Loading molecular dataset: {input_file}")
    df = pd.read_csv(input_file)
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
    
    for smiles in tqdm(df[PreprocessConfig.SMILES_COLUMN], desc="Processing"):
        # Convert to SELFIES
        selfies = smiles_to_selfies(smiles)
        
        if selfies is None:
            stats['failed_conversion'] += 1
            continue
        
        # Validate length
        if not validate_selfies(
            selfies,
            PreprocessConfig.MAX_SEQUENCE_LENGTH,
            PreprocessConfig.MIN_SEQUENCE_LENGTH
        ):
            try:
                length = len(list(sf.split_selfies(selfies)))
                if length > PreprocessConfig.MAX_SEQUENCE_LENGTH:
                    stats['too_long'] += 1
                elif length < PreprocessConfig.MIN_SEQUENCE_LENGTH:
                    stats['too_short'] += 1
                else:
                    stats['invalid'] += 1
            except Exception:
                stats['invalid'] += 1
            continue
        
        valid_selfies.append(selfies)
        stats['converted'] += 1
    
    return valid_selfies, stats


def save_processed_data(
    selfies_list: List[str],
    output_file: Path,
    logger: logging.Logger
) -> None:
    """
    Save processed SELFIES data to CSV file.
    
    Args:
        selfies_list: List of valid SELFIES strings
        output_file: Path to output CSV file
        logger: Logger instance
    """
    logger.info(f"Saving processed data to: {output_file}")
    
    df = pd.DataFrame({'selfies': selfies_list})
    df.to_csv(output_file, index=False)
    
    logger.info(f"Successfully saved {len(selfies_list):,} molecules")


def save_vocabulary(
    vocab_set: Set[str],
    vocab_file: Path,
    logger: logging.Logger
) -> List[str]:
    """
    Build and save vocabulary JSON file.
    
    Args:
        vocab_set: Set of unique tokens
        vocab_file: Path to output vocabulary file
        logger: Logger instance
        
    Returns:
        Final vocabulary list with special tokens
    """
    logger.info("Building vocabulary...")
    
    # Combine special tokens with sorted vocabulary
    final_vocab = PreprocessConfig.SPECIAL_TOKENS + sorted(list(vocab_set))
    
    logger.info(f"Vocabulary size: {len(final_vocab):,} tokens")
    logger.info(f"  - Special tokens: {len(PreprocessConfig.SPECIAL_TOKENS)}")
    logger.info(f"  - Molecular tokens: {len(vocab_set):,}")
    
    # Save to JSON
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(final_vocab, f, indent=2)
    
    logger.info(f"Vocabulary saved to: {vocab_file}")
    
    return final_vocab


def print_statistics(stats: Dict[str, int], logger: logging.Logger) -> None:
    """
    Print detailed processing statistics.
    
    Args:
        stats: Dictionary of processing statistics
        logger: Logger instance
    """
    logger.info("\n" + "="*60)
    logger.info("Processing Statistics")
    logger.info("="*60)
    logger.info(f"Total molecules:           {stats['total']:>10,}")
    logger.info(f"Successfully converted:    {stats['converted']:>10,} ({stats['converted']/stats['total']*100:.2f}%)")
    logger.info(f"Failed conversion:         {stats['failed_conversion']:>10,}")
    logger.info(f"Sequences too long:        {stats['too_long']:>10,}")
    logger.info(f"Sequences too short:       {stats['too_short']:>10,}")
    logger.info(f"Invalid/corrupted:         {stats['invalid']:>10,}")
    logger.info("="*60 + "\n")


def print_sample_data(
    selfies_list: List[str],
    vocab: List[str],
    logger: logging.Logger,
    n_samples: int = 3
) -> None:
    """
    Print sample molecules and vocabulary tokens.
    
    Args:
        selfies_list: List of SELFIES strings
        vocab: Vocabulary list
        logger: Logger instance
        n_samples: Number of samples to display
    """
    logger.info("\n" + "="*60)
    logger.info("Sample Molecules (SELFIES)")
    logger.info("="*60)
    
    for i, selfies in enumerate(selfies_list[:n_samples], 1):
        tokens = list(sf.split_selfies(selfies))
        logger.info(f"\nMolecule {i}:")
        logger.info(f"  Length: {len(tokens)} tokens")
        logger.info(f"  SELFIES: {selfies[:100]}{'...' if len(selfies) > 100 else ''}")
    
    logger.info("\n" + "="*60)
    logger.info("Vocabulary Sample (First 20 tokens)")
    logger.info("="*60)
    logger.info(f"{vocab[:20]}")
    logger.info("="*60 + "\n")


def preprocess_pipeline() -> bool:
    """
    Execute complete preprocessing pipeline.
    
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger()
    
    logger.info("="*60)
    logger.info("MUG Data Preprocessing Pipeline")
    logger.info("="*60)
    
    # Validate input
    if not validate_input_file(PreprocessConfig.INPUT_FILE, logger):
        return False
    
    # Create output directory
    PreprocessConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {PreprocessConfig.OUTPUT_DIR}")
    
    try:
        # Process molecules
        selfies_list, stats = process_molecules(PreprocessConfig.INPUT_FILE, logger)
        
        if not selfies_list:
            logger.error("No valid molecules after processing. Please check input data.")
            return False
        
        # Print statistics
        print_statistics(stats, logger)
        
        # Extract vocabulary
        vocab_set = extract_vocabulary(selfies_list, logger)
        
        # Save processed data
        save_processed_data(selfies_list, PreprocessConfig.OUTPUT_FILE, logger)
        
        # Save vocabulary
        final_vocab = save_vocabulary(vocab_set, PreprocessConfig.VOCAB_FILE, logger)
        
        # Print samples
        print_sample_data(selfies_list, final_vocab, logger)
        
        logger.info("✅ Preprocessing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point for preprocessing script."""
    success = preprocess_pipeline()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
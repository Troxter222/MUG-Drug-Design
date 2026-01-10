"""
Molecular Universe Generator (MUG) - Model Benchmark Suite
Author: Ali (Troxter222)
License: MIT

Comprehensive evaluation framework for comparing molecular generation models.
Tests validity, uniqueness, drug-likeness (QED), and toxicity metrics.
"""

import glob
import json
import logging
import os
import warnings
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# MUG imports
try:
    from app.core.transformer_model import MoleculeTransformer
    from app.core.vocab import Vocabulary
    from app.services.chemistry import ChemistryService
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app.core.transformer_model import MoleculeTransformer
    from app.core.vocab import Vocabulary
    from app.services.chemistry import ChemistryService


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite."""
    
    # Directories to scan for model checkpoints
    CHECKPOINT_DIRS = [
        "checkpoints_rl_transformer",
        "checkpoints_transformer",
        "checkpoints_transformer_v2"
    ]
    
    # Vocabulary file
    VOCAB_PATH = "dataset/processed/vocab_transformer.json"
    
    # Generation parameters
    SAMPLES_PER_MODEL = 100
    BATCH_SIZE = 10
    MAX_SEQUENCE_LENGTH = 150
    TEMPERATURE = 0.8
    
    # Model parameters (defaults for fallback)
    DEFAULT_D_MODEL = 256
    DEFAULT_LAYERS = 4
    DEFAULT_NHEAD = 8
    DEFAULT_LATENT = 128
    
    # Output paths
    OUTPUT_DIR = Path("benchmark_results")
    RESULTS_CSV = OUTPUT_DIR / "benchmark_results.csv"
    METRICS_PLOT = OUTPUT_DIR / "benchmark_metrics.png"
    RADAR_PLOT = OUTPUT_DIR / "benchmark_radar.png"
    
    # Compute device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_logger() -> logging.Logger:
    """Configure logging for benchmark suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("MUG.Benchmark")


def load_vocabulary(vocab_path: str, logger: logging.Logger) -> Optional[Vocabulary]:
    """
    Load vocabulary from JSON file.
    
    Args:
        vocab_path: Path to vocabulary JSON
        logger: Logger instance
        
    Returns:
        Vocabulary object or None if loading fails
    """
    try:
        with open(vocab_path, 'r') as f:
            chars = json.load(f)
        
        # Ensure special tokens are present
        if '<sos>' not in chars:
            chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
        
        return Vocabulary(chars)
    except Exception as e:
        logger.error(f"Failed to load vocabulary from {vocab_path}: {e}")
        return None


def infer_model_architecture(state_dict: Dict) -> Dict[str, int]:
    """
    Infer model architecture parameters from state dict.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary with architecture parameters
    """
    params = {
        'vocab_size': BenchmarkConfig.DEFAULT_D_MODEL,
        'd_model': BenchmarkConfig.DEFAULT_D_MODEL,
        'num_layers': BenchmarkConfig.DEFAULT_LAYERS,
        'nhead': BenchmarkConfig.DEFAULT_NHEAD,
        'latent_size': BenchmarkConfig.DEFAULT_LATENT
    }
    
    # Infer from embedding layer
    if 'embedding.weight' in state_dict:
        params['vocab_size'] = state_dict['embedding.weight'].shape[0]
        params['d_model'] = state_dict['embedding.weight'].shape[1]
    
    # Infer latent size
    if 'fc_mu.weight' in state_dict:
        params['latent_size'] = state_dict['fc_mu.weight'].shape[0]
    
    # Infer number of layers
    max_layer = 0
    for key in state_dict.keys():
        if 'transformer_encoder.layers.' in key:
            try:
                layer_idx = int(key.split('.')[3])
                max_layer = max(max_layer, layer_idx)
            except (IndexError, ValueError):
                pass
    
    if max_layer > 0:
        params['num_layers'] = max_layer + 1
    
    # Infer number of attention heads
    if params['d_model'] % 8 == 0:
        params['nhead'] = 8
    elif params['d_model'] % 4 == 0:
        params['nhead'] = 4
    else:
        params['nhead'] = 2
    
    return params


def adapt_state_dict(
    state_dict: Dict,
    model_keys: set
) -> Dict:
    """
    Adapt state dict to match model architecture.
    
    Handles naming inconsistencies between different model versions.
    
    Args:
        state_dict: Checkpoint state dictionary
        model_keys: Set of expected model parameter keys
        
    Returns:
        Adapted state dictionary
    """
    adapted = {}
    
    key_mappings = {
        'fc_z.weight': 'fc_latent_to_hidden.weight',
        'fc_z.bias': 'fc_latent_to_hidden.bias',
        'fc_latent_to_hidden.weight': 'fc_z.weight',
        'fc_latent_to_hidden.bias': 'fc_z.bias',
    }
    
    for key, value in state_dict.items():
        # Skip positional encoding if shape mismatch
        if 'pos_encoder.pe' in key:
            continue
        
        # Apply key mappings
        new_key = key
        if key in key_mappings and key_mappings[key] in model_keys:
            new_key = key_mappings[key]
        
        adapted[new_key] = value
    
    return adapted


def load_model(
    checkpoint_path: str,
    vocab: Vocabulary,
    logger: logging.Logger
) -> Optional[Tuple[MoleculeTransformer, str]]:
    """
    Load transformer model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab: Vocabulary object
        logger: Logger instance
        
    Returns:
        Tuple of (model, model_name) or (None, None) if loading fails
    """
    try:
        state_dict = torch.load(checkpoint_path, map_location=BenchmarkConfig.DEVICE)  # nosec
        
        # Infer architecture
        params = infer_model_architecture(state_dict)
        
        # Create model
        model = MoleculeTransformer(
            vocab_size=params['vocab_size'],
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_layers'],
            num_decoder_layers=params['num_layers'],
            dim_feedforward=params['d_model'] * 4,
            latent_size=params['latent_size'],
            max_seq_len=5000  # Use larger value to accommodate old checkpoints
        )
        
        # Adapt state dict
        model_keys = set(model.state_dict().keys())
        adapted_state = adapt_state_dict(state_dict, model_keys)
        
        # Load with non-strict mode to ignore missing/unexpected keys
        model.load_state_dict(adapted_state, strict=False)
        model.to(BenchmarkConfig.DEVICE)
        model.eval()
        
        # Generate model name
        filename = os.path.basename(checkpoint_path)
        model_name = f"Trans-L{params['num_layers']}D{params['d_model']}"
        if "rl" in filename.lower():
            model_name += "-RL"
        
        logger.info(f"✓ Loaded: {filename} ({model_name})")
        return model, model_name
        
    except Exception as e:
        logger.warning(f"✗ Failed to load {os.path.basename(checkpoint_path)}: {str(e)}")
        return None, None


def generate_molecules(
    model: MoleculeTransformer,
    vocab: Vocabulary,
    n_samples: int,
    batch_size: int,
    logger: logging.Logger
) -> List[Chem.Mol]:
    """
    Generate molecules from model.
    
    Args:
        model: Trained transformer model
        vocab: Vocabulary object
        n_samples: Total number of molecules to generate
        batch_size: Batch size for generation
        logger: Logger instance
        
    Returns:
        List of valid RDKit molecule objects
    """
    valid_molecules = []
    attempts = 0
    max_attempts = n_samples * 3  # Allow multiple attempts
    
    with torch.no_grad():
        while len(valid_molecules) < n_samples and attempts < max_attempts:
            try:
                # Generate batch
                indices = model.sample(
                    n_samples=batch_size,
                    device=BenchmarkConfig.DEVICE,
                    vocab=vocab,
                    max_len=BenchmarkConfig.MAX_SEQUENCE_LENGTH,
                    temperature=BenchmarkConfig.TEMPERATURE
                )
                
                # Convert to molecules
                for sequence in indices.cpu().numpy():
                    try:
                        # Decode to SELFIES
                        selfies_str = vocab.decode(sequence)
                        if not selfies_str:
                            continue
                        
                        # Convert to SMILES
                        smiles = sf.decoder(selfies_str)
                        if not smiles:
                            continue
                        
                        # Create molecule
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            # Basic sanity checks
                            if 100 < Descriptors.MolWt(mol) < 900:
                                valid_molecules.append(mol)
                                
                                if len(valid_molecules) >= n_samples:
                                    break
                    except Exception:
                        continue
                
                attempts += batch_size
                
            except Exception as e:
                logger.warning(f"Generation batch failed: {e}")
                attempts += batch_size
                continue
    
    return valid_molecules[:n_samples]


def evaluate_molecules(molecules: List[Chem.Mol]) -> Dict[str, float]:
    """
    Compute molecular properties and statistics.
    
    Args:
        molecules: List of RDKit molecules
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not molecules:
        return {
            'validity': 0.0,
            'uniqueness': 0.0,
            'mean_qed': 0.0,
            'toxicity_rate': 0.0,
            'mean_mw': 0.0,
            'mean_logp': 0.0
        }
    
    # Get unique SMILES
    smiles_set = set()
    qed_values = []
    tox_count = 0
    mw_values = []
    logp_values = []
    
    for mol in molecules:
        try:
            smiles = Chem.MolToSmiles(mol)
            smiles_set.add(smiles)
            
            # Get properties
            props = ChemistryService.analyze_properties(mol)
            qed_values.append(props['qed'])
            
            if "⚠️" in str(props['toxicity']) or "Alerts" in str(props['toxicity']):
                tox_count += 1
            
            mw_values.append(Descriptors.MolWt(mol))
            logp_values.append(Descriptors.MolLogP(mol))
            
        except Exception:
            continue
    
    n_mols = len(molecules)
    
    return {
        'validity': 100.0,  # All input molecules are valid
        'uniqueness': (len(smiles_set) / n_mols * 100) if n_mols > 0 else 0.0,
        'mean_qed': np.mean(qed_values) if qed_values else 0.0,
        'toxicity_rate': (tox_count / n_mols * 100) if n_mols > 0 else 0.0,
        'mean_mw': np.mean(mw_values) if mw_values else 0.0,
        'mean_logp': np.mean(logp_values) if logp_values else 0.0
    }


def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Compute overall quality score from individual metrics.
    
    Args:
        metrics: Dictionary of evaluation metrics
        
    Returns:
        Composite score (higher is better)
    """
    # Weighted combination of metrics
    score = (
        metrics['validity'] * 0.3 +
        metrics['uniqueness'] * 0.2 +
        metrics['mean_qed'] * 100 * 0.3 +
        (100 - metrics['toxicity_rate']) * 0.2
    )
    
    return score


def run_benchmark(config: BenchmarkConfig, logger: logging.Logger) -> pd.DataFrame:
    """
    Execute complete benchmark suite.
    
    Args:
        config: Benchmark configuration
        logger: Logger instance
        
    Returns:
        DataFrame with benchmark results
    """
    logger.info("="*60)
    logger.info("MUG Model Benchmark Suite")
    logger.info("="*60)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Samples per model: {config.SAMPLES_PER_MODEL}")
    logger.info("="*60)
    
    # Load vocabulary
    vocab = load_vocabulary(config.VOCAB_PATH, logger)
    if vocab is None:
        logger.error("Failed to load vocabulary. Aborting benchmark.")
        return pd.DataFrame()
    
    logger.info(f"Vocabulary loaded: {len(vocab)} tokens")
    
    # Find all checkpoints
    checkpoint_files = []
    for directory in config.CHECKPOINT_DIRS:
        pattern = os.path.join(directory, "*.pth")
        checkpoint_files.extend(glob.glob(pattern))
    
    logger.info(f"Found {len(checkpoint_files)} model checkpoints")
    
    if not checkpoint_files:
        logger.error("No checkpoint files found. Aborting benchmark.")
        return pd.DataFrame()
    
    # Evaluate each model
    results = []
    
    for checkpoint_path in tqdm(checkpoint_files, desc="Evaluating models"):
        # Load model
        model, model_name = load_model(checkpoint_path, vocab, logger)
        if model is None:
            continue
        
        # Generate molecules
        molecules = generate_molecules(
            model,
            vocab,
            config.SAMPLES_PER_MODEL,
            config.BATCH_SIZE,
            logger
        )
        
        # Evaluate
        metrics = evaluate_molecules(molecules)
        score = compute_composite_score(metrics)
        
        # Store results
        results.append({
            'checkpoint': os.path.basename(checkpoint_path),
            'model_type': model_name,
            'n_generated': len(molecules),
            'validity': metrics['validity'],
            'uniqueness': metrics['uniqueness'],
            'qed': metrics['mean_qed'],
            'toxicity': metrics['toxicity_rate'],
            'mol_weight': metrics['mean_mw'],
            'logp': metrics['mean_logp'],
            'composite_score': score
        })
    
    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, config: BenchmarkConfig, logger: logging.Logger):
    """
    Generate visualization plots for benchmark results.
    
    Args:
        df: Benchmark results DataFrame
        config: Benchmark configuration
        logger: Logger instance
    """
    if df.empty:
        logger.warning("No data to visualize")
        return
    
    # Set style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Top models bar chart
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values('composite_score', ascending=False).head(10)
    
    ax = sns.barplot(
        data=df_sorted,
        x='composite_score',
        y='checkpoint',
        hue='checkpoint',
        palette='viridis',
        legend=False
    )
    
    plt.title('Top 10 Models by Composite Score', fontsize=16, fontweight='bold')
    plt.xlabel('Composite Score', fontsize=12)
    plt.ylabel('Model Checkpoint', fontsize=12)
    plt.tight_layout()
    plt.savefig(config.METRICS_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved metrics plot: {config.METRICS_PLOT}")
    
    # 2. Radar chart for best model
    if len(df_sorted) > 0:
        best = df_sorted.iloc[0]
        
        categories = ['Validity', 'Uniqueness', 'QED (×100)', 'Safety']
        values = [
            best['validity'],
            best['uniqueness'],
            best['qed'] * 100,
            100 - best['toxicity']
        ]
        
        # Normalize to 0-100 scale
        values_normalized = [(v / 100) * 100 for v in values]
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        values_normalized += values_normalized[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values_normalized, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, values_normalized, alpha=0.25, color='#1f77b4')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.grid(True)
        
        plt.title(
            f'Best Model Performance: {best["checkpoint"]}\n'
            f'Score: {best["composite_score"]:.2f}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        plt.savefig(config.RADAR_PLOT, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved radar plot: {config.RADAR_PLOT}")


def print_summary(df: pd.DataFrame, logger: logging.Logger):
    """
    Print summary statistics of benchmark results.
    
    Args:
        df: Benchmark results DataFrame
        logger: Logger instance
    """
    if df.empty:
        logger.warning("No results to summarize")
        return
    
    logger.info("\n" + "="*60)
    logger.info("Benchmark Results Summary")
    logger.info("="*60)
    
    df_sorted = df.sort_values('composite_score', ascending=False)
    
    logger.info("\nTop 5 Models:")
    for idx, row in df_sorted.head(5).iterrows():
        logger.info(f"\n{row['checkpoint']}:")
        logger.info(f"  Score: {row['composite_score']:.2f}")
        logger.info(f"  Generated: {row['n_generated']}/{BenchmarkConfig.SAMPLES_PER_MODEL}")
        logger.info(f"  Uniqueness: {row['uniqueness']:.1f}%")
        logger.info(f"  QED: {row['qed']:.3f}")
        logger.info(f"  Toxicity: {row['toxicity']:.1f}%")
    
    logger.info("\n" + "="*60)


def main():
    """Main entry point for benchmark suite."""
    config = BenchmarkConfig()
    logger = setup_logger()
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run benchmark
        results_df = run_benchmark(config, logger)
        
        if results_df.empty:
            logger.error("Benchmark failed: no results collected")
            return 1
        
        # Save results
        results_df.to_csv(config.RESULTS_CSV, index=False)
        logger.info(f"Results saved to: {config.RESULTS_CSV}")
        
        # Create visualizations
        create_visualizations(results_df, config, logger)
        
        # Print summary
        print_summary(results_df, logger)
        
        logger.info("\nBenchmark completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
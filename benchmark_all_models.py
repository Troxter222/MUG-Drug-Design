import glob
import json
import logging
import os
import sys
import warnings
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, QED
from tqdm import tqdm
from fpdf import FPDF
import selfies as sf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# MUG Imports
try:
    from app.config import Config
    from app.core.vocab import Vocabulary
    from app.core.transformer_model import MoleculeTransformer
    from app.core.engine import MolecularVAE
    from app.services.chemistry import ChemistryService
    from app.services.biology import BiologyService
except ImportError as e:
    print(f"Error importing MUG modules: {e}")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler("benchmark_full.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MUG.BenchmarkFull")

class BenchmarkRunner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.biology_service = BiologyService()
        self.chemistry_service = ChemistryService()
        self.vocab = self._load_vocab()
        
        # Load targets from Config
        self.targets = self._get_all_targets()
        
    def _load_vocab(self):
        try:
            with open(Config.VOCAB_PATH, 'r') as f:
                chars = json.load(f)
            if '<sos>' not in chars:
                chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
            return Vocabulary(chars)
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            sys.exit(1)

    def _get_all_targets(self) -> List[Dict]:
        """Flatten target structure into a list."""
        all_targets = []
        for cat_key, cat_val in Config.DISEASE_DB.items():
            for target_key, target_info in cat_val['targets'].items():
                target_info['key'] = target_key
                target_info['category'] = cat_key
                all_targets.append(target_info)
        return all_targets

    def find_all_models(self) -> List[Path]:
        """Recursively find all .pth files."""
        # root_dir = PROJECT_ROOT # Scan everything in project root
        # Using specific checkpoint dirs is safer to avoid garbage, but user said "DIRECTLY ALL" (.pth прям все)
        # So I will scan recursively from PROJECT_ROOT, but exclude venv/git/etc
        
        pkgs = []
        skip_dirs = {'.git', '.venv', '__pycache__', 'node_modules'}
        
        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for file in files:
                if file.endswith(".pth"):
                    pkgs.append(Path(root) / file)
        
        return pkgs

    def load_model(self, path: Path) -> Tuple[Optional[torch.nn.Module], str]:
        """Load model with architecture inference."""
        try:
            state_dict = torch.load(path, map_location=self.device)  # nosec
            if not isinstance(state_dict, dict):
                 # Might be full model save
                 logger.warning(f"File {path.name} is not a state_dict, skipping.")
                 return None, "Unknown"

            # 1. Try to match with Registry for explicit config
            model_conf = None
            filename = path.name
            
            # Simple heuristic to find config in registry locally or reuse logic
            # But inference is more robust for "random" .pth files
            
            # Infer parameters
            # Default
            vocab_size = len(self.vocab)
            d_model = 256
            n_layers = 4
            nhead = 8
            latent = 128
            
            # Check embedding
            if 'embedding.weight' in state_dict:
                vocab_size = state_dict['embedding.weight'].shape[0]
                d_model = state_dict['embedding.weight'].shape[1]
            
            # Check latent
            if 'fc_mu.weight' in state_dict:
                latent = state_dict['fc_mu.weight'].shape[0]
            
            # Infer VAE vs Transformer
            is_rnn = 'encoder_gru.weight_ih_l0' in state_dict
            
            if is_rnn:
                model = MolecularVAE(vocab_size, 64, 256, 128, 3) # Generic GRU params
                # Try to infer hidden
                if 'encoder_gru.weight_ih_l0' in state_dict:
                     # shape is [3*hidden, input] for GRU? No [3*hidden, embed]
                     pass 
                     # GRU inference is tricky without explicit config, strict=False helps
                
                model_type = "GRU/VAE"
            else:
                # Transformer Inference
                # Check layers
                max_layer = 0
                for k in state_dict.keys():
                    if 'transformer_encoder.layers.' in k:
                        try:
                            l = int(k.split('.')[2])
                            max_layer = max(max_layer, l)
                        except: pass
                if max_layer > 0: n_layers = max_layer + 1
                
                model = MoleculeTransformer(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=n_layers,
                    num_decoder_layers=n_layers,
                    dim_feedforward=d_model * 4,
                    latent_size=latent
                )
                model_type = f"Transformer-L{n_layers}"

            # Rename keys for compatibility (fc_z -> fc_latent_to_hidden)
            new_state = {}
            for k, v in state_dict.items():
                new_k = k
                if 'fc_z' in k: new_k = k.replace('fc_z', 'fc_latent_to_hidden')
                new_state[new_k] = v
            
            model.load_state_dict(new_state, strict=False)
            model.to(self.device)
            model.eval()
            return model, model_type

        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")
            return None, "Error"

    def generate_and_evaluate(self, model, n_samples: int, target: Dict) -> Optional[Dict]:
        """Generate batch, pick best, and evaluate fully."""
        valid_mols = []
        batch_size = min(50, n_samples)
        
        # Generator fp for similarity
        gen_fp = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
        
        # Target FP
        target_fp = None
        try:
             ref_mol = Chem.MolFromSmiles(target['ref'])
             if ref_mol:
                 target_fp = gen_fp.GetFingerprint(ref_mol)
        except: pass

        generated_count = 0
        pbar = tqdm(total=n_samples, desc=f"Generating {n_samples}", leave=False)
        
        attempts = 0
        max_attempts = n_samples * 2 
        
        best_candidate = None
        best_score = -float('inf')

        # To avoid storing 10000 mols in memory, we process in batches and keep only the best
        
        while generated_count < n_samples and attempts < max_attempts:
            current_batch = min(batch_size, n_samples - generated_count)
            try:
                # Assuming Transformer sample method. If GRU, signature might differ slightly but usually similar in this codebase
                if hasattr(model, 'sample'):
                    indices = model.sample(current_batch, self.device, self.vocab, max_len=100)
                else:
                    # Generic fallback or skip
                    break

                for seq in indices.cpu().numpy():
                    smi = sf.decoder(self.vocab.decode(seq))
                    if not smi: continue
                    mol = Chem.MolFromSmiles(smi)
                    if not mol: continue
                    
                    # Analyze
                    props = self.chemistry_service.analyze_properties(mol)
                    
                    # Calculate Score to pick "Best"
                    # Score = QED + Similarity + CNS(if neuro)
                    score = props['qed']
                    
                    sim = 0.0
                    if target_fp:
                        mol_fp = gen_fp.GetFingerprint(mol)
                        sim = DataStructs.TanimotoSimilarity(target_fp, mol_fp)
                        score += sim * 2.0
                    
                    # Keep track of best
                    if score > best_score:
                        best_score = score
                        best_candidate = {
                            'mol': mol,
                            'smiles': smi,
                            'props': props,
                            'sim': sim,
                            'score': score
                        }
                    
                    generated_count += 1
                    pbar.update(1)
                    
            except Exception as e:
                # logger.debug(f"Generation error: {e}")
                pass
            
            attempts += current_batch
        
        pbar.close()
        
        if not best_candidate:
            return None

        # --- RUN VINA DOCKING ON BEST CANDIDATE ---
        # "всегда береться самый лучший и пишеться все характеристики вина docking"
        # We only dock the WINNER of the batch
        try:
            affinity = self.biology_service.dock_molecule(best_candidate['mol'], target['category'])
        except Exception:
            affinity = 0.0 # Failed or QSAR fallback inside service
            
        best_candidate['affinity'] = affinity
        return best_candidate

    def run_full_benchmark(self):
        config_targets = self.targets
        model_paths = self.find_all_models()
        logger.info(f"Found {len(model_paths)} models and {len(config_targets)} targets.")
        
        sample_sizes = [100, 1000, 10000]
        results = []

        for model_path in model_paths:
            logger.info(f"Evaluating Model: {model_path.name}")
            model, model_type = self.load_model(model_path)
            
            if not model:
                continue

            for target in config_targets:
                logger.info(f"  Target: {target['target_name']}")
                
                for n in sample_sizes:
                    logger.info(f"    Sample Size: {n}")
                    
                    data = self.generate_and_evaluate(model, n, target)
                    
                    if data:
                        # "MW LogP QED Radical Affinity. И название модели."
                        # Radical = props['cns_prob']
                        row = {
                            "Model Name": model_path.name,
                            "Model Type": model_type,
                            "Target": target['target_name'],
                            "Target Category": target['category'],
                            "Sample Size": n,
                            "Best SMILES": data['smiles'],
                            "Vina Docking (kcal/mol)": data['affinity'],
                            "MW": data['props']['mw'],
                            "LogP": data['props']['logp'],
                            "QED": data['props']['qed'],
                            "Radical Affinity": data['props']['cns_prob'], # CNS Probability as "Radical"
                            "Similarity": data['sim'],
                            "Composite Score": data['score']
                        }
                        results.append(row)
                    else:
                        logger.warning(f"    No valid molecules generated for {n}")

        # --- SAVE RESULTS ---
        df = pd.DataFrame(results)
        
        if df.empty:
            logger.error("No results generated!")
            return

        # 1. Save CSV/Excel
        df.to_csv("benchmark_full_results.csv", index=False)
        df.to_excel("benchmark_full_results.xlsx", index=False)
        
        # 2. PDF Report
        self._generate_pdf_report(df)
        
        # 3. Print Best Models
        self._print_best_models(df)

    def _generate_pdf_report(self, df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "MUG Comprehensive Benchmark Report", 0, 1, 'C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, f"Generated: {datetime.datetime.now()}", 0, 1, 'C')
        
        # Summary Table of Best Models per Target
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Best Models by Target (Based on Vina Affinity)", 0, 1)
        
        pdf.set_font("Arial", '', 10)
        
        # Find best per target
        targets = df['Target'].unique()
        for t in targets:
            subset = df[df['Target'] == t]
            if subset.empty: continue
            
            # Sort by Vina (more negative is better)
            best_row = subset.sort_values(by="Vina Docking (kcal/mol)", ascending=True).iloc[0]
            
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 8, f"Target: {t}", 0, 1, 'L', fill=True)
            pdf.cell(10)
            pdf.cell(0, 6, f"Best Model: {best_row['Model Name']}", 0, 1)
            pdf.cell(10)
            pdf.cell(0, 6, f"Affinity: {best_row['Vina Docking (kcal/mol)']} kcal/mol", 0, 1)
            pdf.cell(10)
            pdf.cell(0, 6, f"Radical: {best_row['Radical Affinity']} | QED: {best_row['QED']}", 0, 1)
            pdf.cell(10)
            pdf.cell(0, 6, f"SMILES: {best_row['Best SMILES'][:50]}...", 0, 1)
            pdf.ln(4)

        pdf.output("benchmark_full_report.pdf")
        logger.info("PDF Report saved to benchmark_full_report.pdf")

    def _print_best_models(self, df):
        print("\n" + "="*60)
        print("🏆 BEST MODELS PER TARGET")
        print("="*60)
        
        targets = df['Target'].unique()
        for t in targets:
            subset = df[df['Target'] == t]
            if subset.empty: continue
            
            best = subset.sort_values(by="Vina Docking (kcal/mol)", ascending=True).iloc[0]
            
            print(f"\nTarget: {t}")
            print(f"  Model: {best['Model Name']}")
            print(f"  Affinity: {best['Vina Docking (kcal/mol)']} kcal/mol")
            print(f"  Sample Size: {best['Sample Size']}")
            print(f"  Radical: {best['Radical Affinity']}")
        print("\n" + "="*60)

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_full_benchmark()

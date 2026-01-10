"""
MUG REST API Service
Powered by FastAPI
"""

import sys
import os
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from rdkit import Chem
import selfies as sf

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from app.config import Config
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService
from app.services.toxicity import ToxicityService
from app.services.retrosynthesis import RetrosynthesisService

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MUG-API")

# --- Global State ---
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML models on startup."""
    logger.info("⏳ Loading MUG Engine...")
    
    # 1. Load Vocab
    try:
        with open(Config.VOCAB_PATH, 'r') as f:
            import json
            chars = json.load(f)
        if '<sos>' not in chars:
            chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
        vocab = Vocabulary(chars)
        ml_resources['vocab'] = vocab
    except Exception as e:
        logger.error(f"Vocab load failed: {e}")
        sys.exit(1)

    # 2. Load Model
    try:
        # Loading the best current model (e.g., Alzheimer or Universal)
        model_key = "trans_v2" # Default base
        conf = Config.MODEL_REGISTRY.get(model_key)
        path = Config.BASE_DIR / conf["path"]
        
        state_dict = torch.load(path, map_location=Config.DEVICE) # nosec
        
        # Architecture params (Must match training)
        params = conf.get("params", {})
        
        model = MoleculeTransformer(
            vocab_size=len(vocab),
            d_model=params.get("embed", 256),
            nhead=params.get("nhead", 8),
            num_encoder_layers=params.get("layers", 4),
            num_decoder_layers=params.get("layers", 4),
            dim_feedforward=params.get("hidden", 1024),
            latent_size=128
        )
        
        # Handle state dict compatibility
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace('fc_z', 'fc_latent_to_hidden')
            new_state[new_k] = v
            
        model.load_state_dict(new_state, strict=False)
        model.to(Config.DEVICE)
        model.eval()
        
        ml_resources['model'] = model
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        sys.exit(1)

    # 3. Load Services
    ml_resources['tox'] = ToxicityService()
    ml_resources['bio'] = BiologyService()
    
    logger.info("✅ System Online.")
    yield
    # Cleanup if needed
    ml_resources.clear()

# --- API Definition ---
app = FastAPI(
    title="MUG API",
    description="Molecular Universe Generator: De Novo Drug Design Interface",
    version="7.5",
    lifespan=lifespan
)

# --- Pydantic Models ---

class GenerateRequest(BaseModel):
    n_samples: int = Field(default=5, ge=1, le=50, description="Number of molecules to generate")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Sampling creativity")
    max_len: int = Field(default=100, description="Max sequence length")

class EvaluateRequest(BaseModel):
    smiles: str = Field(..., description="Molecule SMILES string")
    target: Optional[str] = Field(None, description="Target key for docking (e.g., 'covid', 'alzheimer')")

class MoleculeResponse(BaseModel):
    smiles: str
    props: Dict[str, Any]
    affinity: Optional[float] = None
    toxicity_risks: List[str]
    synthesis_route: str

# --- Endpoints ---

@app.get("/health", tags=["System"])
def health_check():
    """Check system status."""
    return {"status": "active", "device": str(Config.DEVICE)}

@app.post("/generate", response_model=List[MoleculeResponse], tags=["Inference"])
async def generate_molecules(req: GenerateRequest):
    """
    Generate novel molecules using the Transformer-VAE.
    """
    model = ml_resources['model']
    vocab = ml_resources['vocab']
    
    # Generate
    with torch.no_grad():
        indices = model.sample(
            req.n_samples, 
            Config.DEVICE, 
            vocab, 
            max_len=req.max_len, 
            temperature=req.temperature
        )
    
    results = []
    
    for seq in indices.cpu().numpy():
        try:
            text = vocab.decode(seq)
            smi = sf.decoder(text)
            mol = Chem.MolFromSmiles(smi)
            if not mol: continue
            
            # Analyze
            chem = ChemistryService()
            props = chem.analyze_properties(mol)
            
            results.append({
                "smiles": smi,
                "props": props,
                "affinity": None,
                "toxicity_risks": [],
                "synthesis_route": "Not requested"
            })
        except Exception:
            continue
            
    return results

@app.post("/evaluate", response_model=MoleculeResponse, tags=["Analysis"])
async def evaluate_molecule(req: EvaluateRequest):
    """
    Full analysis of a specific molecule: Properties, Toxicity, Docking (optional).
    """
    mol = Chem.MolFromSmiles(req.smiles)
    if not mol:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    
    # 1. Chemistry
    props = ChemistryService.analyze_properties(mol)
    
    # 2. Toxicity
    tox_service = ml_resources['tox']
    risks = tox_service.predict(mol)
    
    # 3. Docking
    affinity = None
    if req.target:
        bio_service = ml_resources['bio']
        # Try to resolve target category automatically
        cat = "unknown"
        if req.target in ["covid", "viral"]: cat = "viral"
        elif req.target in ["alzheimer", "neuro"]: cat = "neuro"
        elif req.target in ["egfr", "onco"]: cat = "onco"
        
        affinity = bio_service.dock_molecule(mol, cat)
        
    # 4. Retro
    blocks = RetrosynthesisService.get_building_blocks(mol)
    complexity = RetrosynthesisService.assess_complexity(mol)
    route = RetrosynthesisService.describe_synthesis(blocks, complexity)
    
    return {
        "smiles": req.smiles,
        "props": props,
        "affinity": affinity,
        "toxicity_risks": risks,
        "synthesis_route": route
    }

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)  # nosec B104
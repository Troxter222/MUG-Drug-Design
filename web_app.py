import streamlit as st
import torch
import io
import json
from PIL import Image

# Импорты из проекта
from app.config import Config
from app.core.vocab import Vocabulary
from app.core.transformer_model import MoleculeTransformer
from app.core.engine import MolecularVAE
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService
from app.services.toxicity import ToxicityService
from app.services.visualization import VisualizationService
from app.services.retrosynthesis import RetrosynthesisService

import selfies as sf
# --- ИСПРАВЛЕННЫЙ ИМПОРТ RDKit ---
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="MUG: Molecular Universe Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cyberpunk Look
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #C0C0C0;
    }
    .stButton>button {
        color: #4CAF50;
        border-color: #4CAF50;
        background-color: #1E1E1E;
        width: 100%;
        height: 50px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        text-align: center;
    }
    h1, h2, h3 {
        color: #00e5ff !important;
        font-family: 'Courier New', monospace;
    }
    /* Скрываем гамбургер-меню и футер для чистоты */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- CACHED LOADERS ---
@st.cache_resource
def load_resources():
    # 1. Config & Vocab
    model_key = Config.CURRENT_MODEL_KEY
    if model_key not in Config.MODEL_REGISTRY:
        # Fallback если конфиг сбился
        model_key = "trans_v2" # или другой рабочий ключ
        
    conf = Config.MODEL_REGISTRY.get(model_key)
    if not conf:
        st.error(f"Model configuration for '{model_key}' not found.")
        return None, None, None, None

    path = Config.BASE_DIR / conf["vocab"]
    
    try:
        with open(path, 'r') as f:
            chars = json.load(f)
    except FileNotFoundError:
        st.error(f"Vocabulary not found at {path}")
        return None, None, None, None

    required = ['<pad>', '<sos>', '<eos>']
    if '<sos>' not in chars: 
        chars = required + sorted(chars)
    vocab = Vocabulary(chars)
    
    # 2. Model
    model_path = Config.BASE_DIR / conf["path"]
    if not model_path.exists():
        st.error(f"Model checkpoint not found at {model_path}")
        return None, None, None, None
        
    state_dict = torch.load(model_path, map_location=Config.DEVICE)
    
    # Architecture switch
    if conf["type"] == "transformer":
        new_state = {}
        for k,v in state_dict.items():
            if 'fc_z' in k: 
                new_state[k.replace('fc_z', 'fc_latent_to_hidden')] = v
            else: 
                new_state[k] = v
        
        # Determine vocab size from weights
        if 'embedding.weight' in state_dict:
            vocab_size = state_dict['embedding.weight'].shape[0]
        else:
            vocab_size = len(vocab)
            
        model = MoleculeTransformer(vocab_size, 256, 8, 4, 4, 1024, 128)
        model.load_state_dict(new_state, strict=False)
    else:
        # GRU
        vocab_size = state_dict['embedding.weight'].shape[0] if 'embedding.weight' in state_dict else len(vocab)
        model = MolecularVAE(vocab_size, 64, 256, 128, 3)
        model.load_state_dict(state_dict, strict=False)
        
    model.to(Config.DEVICE)
    model.eval()
    
    # 3. Services
    tox_service = ToxicityService()
    
    return model, vocab, tox_service, conf["type"]

# Load resources
try:
    model, vocab, tox_service, model_type = load_resources()
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

if model is None:
    st.warning("System could not initialize. Check logs/paths.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🌌 MUG Control Panel")
st.sidebar.markdown(f"**Core:** `{Config.MODEL_REGISTRY[Config.CURRENT_MODEL_KEY]['name']}`")
st.sidebar.markdown(f"**Device:** `{Config.DEVICE}`")

mode = st.sidebar.radio("Operation Mode:", ["Targeted Design", "Random Exploration"])

target_info = None
target_fp = None
target_cat = ""

if mode == "Targeted Design":
    disease_area = st.sidebar.selectbox("Therapeutic Area", list(Config.DISEASE_DB.keys()))
    
    # Get Targets
    targets = Config.DISEASE_DB[disease_area]['targets']
    # Создаем маппинг имен для селекта
    target_names = {k: v['target_name'] for k, v in targets.items()}
    target_key = st.sidebar.selectbox("Target Protein", list(target_names.keys()), format_func=lambda x: target_names[x])
    
    target_info = targets[target_key]
    target_cat = disease_area
    
    # Prepare Fingerprint
    ref_mol = Chem.MolFromSmiles(target_info['ref'])
    if ref_mol:
        target_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
    
    st.sidebar.success(f"🎯 Locked: {target_info['target_name']}")

# --- MAIN AREA ---
st.title("Molecular Universe Generator")
st.markdown("### AI-Powered De Novo Drug Design Platform")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 🧪 Generation Workbench")
    
    if st.button("🚀 IGNITE GENERATION", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        attempts = 20 if mode == "Targeted Design" else 5
        batch_size = 20
        
        best_score = -1000
        best_cand = None
        
        for i in range(attempts):
            status_text.text(f"⚡ Neural Search Iteration {i+1}/{attempts}...")
            progress_bar.progress(int((i + 1) / attempts * 100))
            
            with torch.no_grad():
                try:
                    if model_type == "transformer":
                        indices = model.sample(batch_size, Config.DEVICE, vocab, max_len=120, temperature=0.7)
                    else:
                        indices = model.sample(batch_size, Config.DEVICE, vocab, max_len=120, temp=0.7)
                except Exception: 
                    continue
                
            # Process Batch
            seqs = indices.cpu().numpy()
            for seq in seqs:
                try:
                    text = vocab.decode(seq)
                    smi = sf.decoder(text)
                    if not smi: 
                        continue
                    mol = Chem.MolFromSmiles(smi)
                    if not mol: 
                        continue
                    
                    mol = ChemistryService.normalize_structure(mol)
                    if Descriptors.MolWt(mol) < 150: 
                        continue
                    
                    props = ChemistryService.analyze_properties(mol)
                    cns_p, cns_w = ChemistryService.validate_cns_rules(mol, target_cat)
                    
                    score = 0
                    aff = 0.0
                    
                    if mode == "Targeted Design":
                        sim = DataStructs.TanimotoSimilarity(target_fp, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                        if props['qed'] > 0.25:
                            aff = BiologyService().dock_molecule(mol, target_cat)
                        
                        score = (sim * 50) + (props['qed'] * 30) + abs(aff * 5) - cns_p
                    else:
                        score = props['qed']
                    
                    if score > best_score:
                        best_score = score
                        # Predict Tox only for best
                        tox_risks = tox_service.predict(mol)
                        best_cand = (mol, Chem.MolToSmiles(mol), props, aff, tox_risks)
                        
                except Exception: 
                    continue
        
        progress_bar.progress(100)
        status_text.text("✅ Search Complete.")
        
        if best_cand:
            mol, smi, props, aff, risks = best_cand
            
            # --- RESULTS DISPLAY ---
            st.markdown("---")
            
            # Top Row: Image & Key Metrics
            c1, c2 = st.columns([1, 1])
            
            with c1:
                img = VisualizationService.draw_cyberpunk(mol)
                st.image(img, caption="AI Generated Structure", width='stretch')
                
            with c2:
                st.markdown("## Generated Candidate")
                st.code(smi, language="text")
                st.markdown(f"**Affinity:** `{aff} kcal/mol`")
                st.markdown(f"**QED:** `{props['qed']}`")
                
                # Metrics Grid
                m1, m2, m3 = st.columns(3)
                m1.metric("MW", props['mw'])
                m2.metric("cLogD (7.4)", props['clogd'])
                m3.metric("TPSA", props['tpsa'])
                
                st.info(f"**CNS Probability:** {props['cns_prob']}")
                
                if risks:
                    st.error("**Risk Flags:** " + ", ".join(risks))
                else:
                    st.success("**AI Toxicology:** No high risks detected")

            # Synthesis
            st.markdown("### 🏗 Synthetic Route")
            blocks = RetrosynthesisService.get_building_blocks(mol)
            complexity = RetrosynthesisService.assess_complexity(mol)
            desc = RetrosynthesisService.describe_synthesis(blocks, complexity)
            st.text(desc)
            
            if blocks:
                st.write("Building Blocks:")
                # Streamlit не умеет напрямую показывать grid image из RDKit, конвертируем
                try:
                    mols = [Chem.MolFromSmiles(b) for b in blocks]
                    img_grid = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200))
                    # Convert to PIL for Streamlit
                    if not isinstance(img_grid, Image.Image):
                         # Иногда RDKit возвращает raw bytes
                         img_grid = Image.open(io.BytesIO(img_grid))
                    st.image(img_grid)
                except Exception:
                    st.write(blocks)

        else:
            st.warning("No viable candidates found. Try again or change strictness.")

with col2:
    st.markdown("#### 📡 System Logs")
    st.text_area("Live Feed", 
                 "System initialized.\n"
                 f"Vocabulary size: {len(vocab)}\n"
                 f"Toxicity models: {len(tox_service.models)}\n"
                 "Ready for task...", 
                 height=400)
    st.markdown("---")
    st.markdown("© 2025 MUG Project")
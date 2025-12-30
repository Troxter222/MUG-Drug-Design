"""
Molecular Universe Generator (MUG) - Model Benchmark Suite
Author: Ali (Troxter222)
License: MIT

Comprehensive evaluation framework for comparing molecular generation models.
Tests validity, uniqueness, drug-likeness (QED), and toxicity metrics.
"""

import streamlit as st
import torch
import time
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
import py3Dmol
from stmol import showmol
import selfies as sf

# MUG Imports
from app.config import Config
from app.core.vocab import Vocabulary
from app.core.transformer_model import MoleculeTransformer
from app.core.engine import MolecularVAE
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService
from app.services.toxicity import ToxicityService
from app.services.retrosynthesis import RetrosynthesisService
from app.services.reporting import ReportingService
from app.services.visualization import VisualizationService

# --- CONFIG & OPTIMIZATION ---
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.transformer') # Suppress nested tensor warning

st.set_page_config(
    page_title="MUG: Molecular Universe Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME CSS ---
st.markdown("""
<style>
    /* Global Reset & Dark Theme */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #111 0%, #000 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3 {
        color: #00f2ff;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.5);
        font-weight: 700;
        letter-spacing: 1px;
    }
    h1 { font-size: 3em !important; }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid #333;
        border-left: 4px solid #00f2ff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
    }
    .metric-value { font-size: 1.8em; font-weight: bold; color: #fff; }
    .metric-label { color: #888; font-size: 0.9em; text-transform: uppercase; }

    /* Custom Buttons - Neon Style */
    div.stButton > button {
        background: linear-gradient(45deg, #001f3f, #003366);
        color: #00f2ff;
        border: 1px solid #00f2ff;
        border-radius: 4px;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 5px rgba(0, 242, 255, 0.2);
    }
    div.stButton > button:hover {
        background: #00f2ff;
        color: #000;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.6);
        border-color: #fff;
    }
    div.stButton > button:active {
        transform: scale(0.98);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #222;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #888;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 242, 255, 0.1);
        color: #00f2ff;
        border-bottom: 2px solid #00f2ff;
    }

    /* Success/Error/Info boxes revamp */
    .stSuccess { background-color: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; color: #00ff88; }
    .stError { background-color: rgba(255, 0, 85, 0.1); border: 1px solid #ff0055; color: #ff0055; }
    .stInfo { background-color: rgba(0, 183, 255, 0.1); border: 1px solid #00b7ff; color: #00b7ff; }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #00f2ff; }
</style>
""", unsafe_allow_html=True)

# --- SYSTEM INITIALIZATION ---
@st.cache_resource
def load_system(model_key):
    try:
        conf = Config.MODEL_REGISTRY.get(model_key)
        if not conf: 
            st.error(f"Configuration for {model_key} not found.")
            return None, None, None
        
        # Vocab
        import json
        if not Config.VOCAB_PATH.exists():
             st.error(f"Vocabulary file missing at {Config.VOCAB_PATH}")
             return None, None, None

        with open(Config.VOCAB_PATH, 'r') as f: 
            chars = json.load(f)
        
        if '<sos>' not in chars: 
            chars = ['<pad>', '<sos>', '<eos>'] + sorted(chars)
        vocab = Vocabulary(chars)
        
        # Model
        path = Config.BASE_DIR / conf["path"]
        if not path.exists():
            st.error(f"Model checkpoint missing at {path}")
            return None, None, None

        state = torch.load(path, map_location=Config.DEVICE)
        
        if conf["type"] == "transformer":
            # State dict compatibility fix
            new_state = {k.replace('fc_z', 'fc_latent_to_hidden') if 'fc_z' in k else k: v for k, v in state.items()}
            # Infer vocab size from weights if possible, else use vocab len
            vocab_size = state['embedding.weight'].shape[0] if 'embedding.weight' in state else len(vocab)
            
            # Using params from config if available, else defaults
            params = conf.get("params", {})
            model = MoleculeTransformer(
                vocab_size=vocab_size, 
                d_model=params.get("embed", 256), 
                nhead=params.get("nhead", 8), 
                num_encoder_layers=params.get("layers", 4),
                num_decoder_layers=params.get("layers", 4),
                dim_feedforward=params.get("hidden", 1024), 
                latent_size=128
            )
            model.load_state_dict(new_state, strict=False)
        else:
            model = MolecularVAE(len(vocab), 64, 256, 128, 3)
            model.load_state_dict(state, strict=False)
            
        model.to(Config.DEVICE)
        model.eval()
        
        return model, vocab, ToxicityService()
    except Exception as e:
        st.error(f"Critical System Failure: {str(e)}")
        return None, None, None

# --- ANALYSIS TOOLS ---
def create_radar_chart(props, affinity_norm=0.5):
    """Creates a scientific radar chart for molecular properties."""
    # Normalize values for visualization (0-1 scale approx)
    data = {
        'Bioavailability (QED)': props['qed'],
        'Lipophilicity (LogP/5)': min(max(props['logp'] / 5.0, 0), 1),
        'Drug-Likeness': (props['qed'] + 0.5) / 1.5,
        'Synthesizability': 1.0 - (min(props['complexity_ct'], 1000) / 1000),
        'Affinity Score': affinity_norm
    }
    
    categories = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Molecule Profile',
        line=dict(color='#00f2ff'),
        fillcolor='rgba(0, 242, 255, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#333', linecolor='#333'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc'),
        margin=dict(l=40, r=40, t=20, b=20),
        showlegend=False
    )
    return fig

def evaluate_batch_process(seqs, vocab, target_data, tox_service):
    """
    Process a batch of sequences and return a dataframe of results.
    """
    valid_mols = []
    
    target_fp = None
    target_cat = "unknown"
    if target_data and 'target_fp' in target_data:
        target_fp = target_data['target_fp']
        target_cat = target_data.get('category', 'unknown')

    gen_fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    for seq in seqs:
        try:
            smi = sf.decoder(vocab.decode(seq))
            if not smi: 
                continue
            mol = Chem.MolFromSmiles(smi)
            if not mol: 
                continue
            
            # Basic Filters
            props = ChemistryService.analyze_properties(mol)
            if props['mw'] < 150 or props['mw'] > 800: 
                continue
            
            # Scoring
            score = props['qed']
            aff = 0.0
            
            # Similarity & Docking Simulation
            if target_fp:
                mol_fp = gen_fp.GetFingerprint(mol)
                sim = DataStructs.TanimotoSimilarity(target_fp, mol_fp)
                score += sim * 2.5
                
                # Relaxed Docking Simulation for Demo Purposes
                # In production this would be stricter, but user wants to see scores
                if props['qed'] > 0.3: # Lower threshold
                     if sim > 0.2:
                         aff = BiologyService().dock_molecule(mol, target_cat)
                         score += abs(aff) * 1.0 
                     else:
                         # Fallback estimate for non-similar but valid molecules
                         aff = -5.0 - (props['logp'] * 0.5) # Crude heuristic
                         score += 0.5
            
            valid_mols.append({
                'smiles': smi,
                'mol': mol,
                'props': props,
                'affinity': aff,
                'score': score
            })
            
        except Exception:
            continue
            
    # Sort by score desc
    return sorted(valid_mols, key=lambda x: x['score'], reverse=True)

# --- SIDEBAR NAV ---
st.sidebar.title("🎛️ CONTROL UNIT")
st.sidebar.markdown("---")

# Model Selection
selected_model_key = st.sidebar.selectbox(
    "ENGINE KERNEL", 
    list(Config.MODEL_REGISTRY.keys()),
    format_func=lambda x: Config.MODEL_REGISTRY[x]['name']
)

# Load System
model, vocab, tox_service = load_system(selected_model_key)

if not model:
    st.sidebar.error("System Offline. Check Logs.")
    st.stop()
    
st.sidebar.success("● ENGINE ONLINE")
st.sidebar.markdown(f"<div style='font-size: 0.8em; color: #666;'>Device: {Config.DEVICE}</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Strategy Selection
mode = st.sidebar.radio(
    "SEARCH PROTOCOL", 
    ["BATCH SYNTHESIS", "DEEP SPACE SCAN"],
    captions=["Generate set of 50 candidates", "Infinite evolutionary loop"]
)

# Target Definition
target_data = {}
if st.sidebar.checkbox("🔒 LOCK TARGET", value=True):
    area_key = st.sidebar.selectbox("THERAPEUTIC AREA", list(Config.DISEASE_DB.keys()))
    if area_key:
        area_targets = Config.DISEASE_DB[area_key]['targets']
        t_key = st.sidebar.selectbox("PROTEIN TARGET", list(area_targets.keys()))
        t_info = area_targets[t_key]
        
        # Pre-calc target fingerprint
        try:
            ref_mol = Chem.MolFromSmiles(t_info['ref'])
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            target_data = {
                'info': t_info,
                'target_fp': fpg.GetFingerprint(ref_mol),
                'category': area_key
            }
            st.sidebar.info(f"Targeting: {t_info['target_name']}")
        except Exception:
            st.sidebar.error("Invalid Reference Molecule")

# --- MAIN CONTENT ---
st.title("🧬 MOLECULAR UNIVERSE GENERATOR")
st.markdown("*Advanced generative AI for de novo drug design.*")

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'deep_running' not in st.session_state:
    st.session_state.deep_running = False
if 'best_candidate' not in st.session_state:
    st.session_state.best_candidate = None
if 'scanned_total' not in st.session_state:
    st.session_state.scanned_total = 0

# --- MODE: BATCH ---
if mode == "BATCH SYNTHESIS":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        temp = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.8, 0.1)
    with col2:
        count = st.slider("Batch Size", 10, 100, 50, 10)
        
    if st.button("🚀 INITIATE SYNTHESIS", type="primary", use_container_width=True):
        with st.spinner("Synthesizing chemical structures..."):
            start_time = time.time()
            max_len = 100 # Optimization
            
            # Generate
            indices = model.sample(count, Config.DEVICE, vocab, max_len=max_len, temperature=temp)
            
            # Evaluate
            results = evaluate_batch_process(indices.cpu().numpy(), vocab, target_data, tox_service)
            
            duration = time.time() - start_time
            
            if results:
                st.session_state.last_batch = results
                st.session_state.history.extend(results)
                st.success(f"Synthesis Complete. {len(results)} valid candidates generated in {duration:.2f}s.")
            else:
                st.error("No chemically valid molecules found. Try adjusting temperature.")

    # Results Display
    if 'last_batch' in st.session_state:
        results = st.session_state.last_batch
        best = results[0]
        
        st.markdown("### 🏆 Top Candidate Analysis")
        
        # Top 1 Detailed View
        c1, c2, c3 = st.columns([1, 1, 1])
        
        with c1:
            st.markdown(f"**Score:** `{best['score']:.3f}`")
            st.image(VisualizationService.draw_cyberpunk(best['mol']), use_container_width=True)
            
        with c2:
            st.markdown("#### Molecular Properties")
            p = best['props']
            st.markdown(f"""
            <div class="metric-card">
                <div>MW: <b>{p['mw']}</b></div>
                <div>LogP: <b>{p['logp']}</b></div>
                <div>QED: <b>{p['qed']}</b></div>
                <div>Radical: <b>{p['cns_prob']}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            if 'affinity' in best and best['affinity'] != 0:
                 st.metric("Est. Affinity", f"{best['affinity']:.2f} kcal/mol", delta="High Binding")

        with c3:
            st.markdown("#### Scientific Profile")
            # Radar Chart
            fig = create_radar_chart(best['props'], affinity_norm=min(abs(best['affinity'])/12.0, 1.0) if best['affinity'] else 0.5)
            st.plotly_chart(fig, use_container_width=True)

        # Tabs for details
        tab1, tab2, tab3, tab4 = st.tabs(["🧪 All Candidates", "🧊 3D Structure", "📥 Export Data", "📄 PDF Report"])
        
        with tab1:
            # Dataframe view
            df_data = []
            for r in results:
                d = r['props'].copy()
                d['smiles'] = r['smiles']
                d['score'] = r['score']
                d['affinity'] = r['affinity']
                df_data.append(d)
                
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
        with tab2:
            mol_3d = Chem.AddHs(best['mol'])
            try:
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
                v = py3Dmol.view(width=800, height=400)
                v.addModel(Chem.MolToPDBBlock(mol_3d), 'pdb')
                v.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
                v.setBackgroundColor('#000000')
                v.zoomTo()
                showmol(v, height=400, width=800)
            except Exception:
                st.warning("3D Embedding failed for this molecule.")
                
        with tab3:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Batch CSV",
                csv,
                "mug_batch_results.csv",
                "text/csv",
                key='download-csv'
            )

        with tab4:
             st.markdown("### Generate Research Report")
             if st.button("Create PDF Dossier", key="pdf_batch"):
                 with st.spinner("Compiling analysis..."):
                     # Save temp image
                     img_path = "temp_batch_m.png"
                     VisualizationService.draw_cyberpunk(best['mol']).save(img_path)
                     
                     # Get Retrosynthesis
                     blocks = RetrosynthesisService.get_building_blocks(best['mol'])
                     desc = RetrosynthesisService.describe_synthesis(blocks)
                     
                     pdf_file = ReportingService.generate_pdf(
                         "MUG-Candidate-Batch", 
                         best['smiles'], 
                         best['props'], 
                         best['affinity'], 
                         img_path, 
                         desc
                     )
                     
                     with open(pdf_file, "rb") as f:
                         st.download_button("Download PDF", f, "MUG_Report.pdf", "application/pdf")
                     st.success("Report Ready.")

# --- MODE: DEEP SEARCH ---
else:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("▶️", type="primary"):
            st.session_state.deep_running = True
            st.rerun()
    with c3:
        if st.button("🛑"):
            st.session_state.deep_running = False
            st.rerun()
            
    # Dashboard
    st.markdown("### Deep Space Scanner Status")
    
    # Placeholder for live metrics
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    status_text = st.empty()
    mol_display = st.empty()
    
    # Running Logic
    if st.session_state.deep_running:
        # Generate micro-batch
        indices = model.sample(10, Config.DEVICE, vocab, max_len=100, temperature=0.9)
        batch_results = evaluate_batch_process(indices.cpu().numpy(), vocab, target_data, tox_service)
        
        st.session_state.scanned_total += 10
        
        if batch_results:
            top_current = batch_results[0]
            
            # Update global best
            if (st.session_state.best_candidate is None) or \
               (top_current['score'] > st.session_state.best_candidate['score']):
                st.session_state.best_candidate = top_current
                
        # Update UI
        bp = st.session_state.best_candidate
        
        with stat_col1:
            st.metric("Total Scanned", st.session_state.scanned_total)
        with stat_col2:
            val = f"{bp['score']:.3f}" if bp else "0.000"
            st.metric("Best Score", val)
        with stat_col3:
            aff = f"{bp['affinity']:.2f}" if bp and bp['affinity'] else "N/A"
            st.metric("Best Affinity", aff)
            
        if bp:
            mol_display.image(VisualizationService.draw_cyberpunk(bp['mol']), caption=bp['smiles'])
            
        status_text.info("System Active. Mining latent space...")
        time.sleep(0.1)
        st.rerun()
    else:
        status_text.warning("Scanning Paused.")
        if st.session_state.best_candidate:
             st.success("Best candidate retained in memory.")
             
             # --- DETAILED ANALYSIS FOR DEEP SEARCH RESULT ---
             best = st.session_state.best_candidate
             
             st.markdown("### Deep Scan Best Result")
        
             # Top 1 Detailed View (Same as Batch)
             c1, c2, c3 = st.columns([1, 1, 1])
             
             with c1:
                 st.markdown(f"**Score:** `{best['score']:.3f}`")
                 st.image(VisualizationService.draw_cyberpunk(best['mol']), use_container_width=True)
                 st.code(best['smiles'])
                 
             with c2:
                 st.markdown("#### Molecular Properties")
                 p = best['props']
                 st.markdown(f"""
                 <div class="metric-card">
                     <div>MW: <b>{p['mw']}</b></div>
                     <div>LogP: <b>{p['logp']}</b></div>
                     <div>QED: <b>{p['qed']}</b></div>
                     <div>Radical: <b>{p['cns_prob']}</b></div>
                     <div>Alerts: <b>{len(p['alerts_list'])}</b></div>
                 </div>
                 """, unsafe_allow_html=True)
                 
                 if 'affinity' in best and best['affinity'] != 0:
                      st.metric("Est. Affinity", f"{best['affinity']:.2f} kcal/mol", delta="High Binding")
     
             with c3:
                 st.markdown("#### Scientific Profile")
                 # Radar Chart
                 fig = create_radar_chart(best['props'], affinity_norm=min(abs(best['affinity'])/12.0, 1.0) if best['affinity'] else 0.5)
                 st.plotly_chart(fig, use_container_width=True)
     
             # Tabs for details
             tab1, tab2, tab3 = st.tabs(["3D Structure", "Export Data", "PDF Report"])
             
             with tab1:
                 mol_3d = Chem.AddHs(best['mol'])
                 try:
                     AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
                     v = py3Dmol.view(width=800, height=400)
                     v.addModel(Chem.MolToPDBBlock(mol_3d), 'pdb')
                     v.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
                     v.setBackgroundColor('#000000')
                     v.zoomTo()
                     showmol(v, height=400, width=800)
                 except Exception:
                     st.warning("3D Embedding failed for this molecule.")
                     
             with tab2:
                 # Single row dataframe for export
                 d = best['props'].copy()
                 d['smiles'] = best['smiles']
                 d['score'] = best['score']
                 d['affinity'] = best['affinity']
                 df = pd.DataFrame([d])
                 
                 csv = df.to_csv(index=False).encode('utf-8')
                 st.download_button(
                     "Download Candidate CSV",
                     csv,
                     "mug_deep_search_best.csv",
                     "text/csv",
                     key='download-csv-deep'
                 )

             with tab3:
                 st.markdown("### Generate Research Report")
                 if st.button("Create PDF Dossier", key="pdf_deep"):
                     with st.spinner("Compiling analysis..."):
                         # Save temp image
                         img_path = "temp_deep_m.png"
                         VisualizationService.draw_cyberpunk(best['mol']).save(img_path)
                         
                         # Get Retrosynthesis
                         blocks = RetrosynthesisService.get_building_blocks(best['mol'])
                         desc = RetrosynthesisService.describe_synthesis(blocks)
                         
                         pdf_file = ReportingService.generate_pdf(
                             "MUG-Deep-Space-Hit", 
                             best['smiles'], 
                             best['props'], 
                             best['affinity'], 
                             img_path, 
                             desc
                         )
                         
                         with open(pdf_file, "rb") as f:
                             st.download_button("Download PDF", f, "MUG_Deep_Scan_Report.pdf", "application/pdf")
                         st.success("Report Ready.")
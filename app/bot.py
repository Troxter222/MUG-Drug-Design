"""
Molecular Universe Generator (MUG) - Application Controller
Version: 7.4 (Deep Search + AI Toxicology)
License: MIT
"""

import io
import json
import logging
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Tuple, List

import selfies as sf
import telebot
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, Draw
from telebot import types

# --- Project Imports ---
from app.config import Config
from app.core.engine import MolecularVAE
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.biology import BiologyService
from app.services.reporting import ReportingService
from app.services.chemistry import ChemistryService
from app.services.linguistics import LinguisticsService
from app.services.retrosynthesis import RetrosynthesisService
from app.services.visualization import VisualizationService
from app.services.toxicity import ToxicityService

# --- Logging Setup ---
def setup_logger() -> logging.Logger:
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s')
    
    file_handler = RotatingFileHandler(Config.LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.getLogger("rdkit").setLevel(logging.WARNING)
    rdBase.DisableLog('rdApp.*')
    
    return logging.getLogger("MUG.Controller")

logger = setup_logger()

# --- Helper Functions ---

def load_vocabulary(path: str) -> Vocabulary:
    try:
        full_path = Config.BASE_DIR / path
        with open(full_path, 'r') as f:
            chars = json.load(f)
        
        required_tokens = ['<pad>', '<sos>', '<eos>']
        if '<sos>' not in chars:
            chars = required_tokens + sorted(chars)
        
        logger.info(f"Vocabulary loaded: {len(chars)} tokens")
        return Vocabulary(chars)
    except Exception as e:
        logger.error(f"Failed to load vocabulary: {e}")
        # Fallback to default if custom fails
        if path != Config.VOCAB_FILENAME:
            return load_vocabulary(Config.VOCAB_FILENAME)
        sys.exit(1)

def load_model_instance(model_key: str, vocab: Vocabulary):
    """Factory method to load specific model architecture."""
    if model_key not in Config.MODEL_REGISTRY:
        logger.error(f"Model key '{model_key}' not found in registry.")
        return None

    conf = Config.MODEL_REGISTRY[model_key]
    path = Config.BASE_DIR / conf["path"]
    params = conf["params"]
    
    if not path.exists():
        logger.error(f"Checkpoint not found: {path}")
        return None

    logger.info(f"Loading model: {conf['name']} ({conf['type']})...")
    
    try:
        state_dict = torch.load(path, map_location=Config.DEVICE)
        
        # Determine architecture
        if conf["type"] == "transformer":
            # Fix legacy keys if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'fc_z' in k:
                    new_state_dict[k.replace('fc_z', 'fc_latent_to_hidden')] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

            # Check vocab dimension
            vocab_size = state_dict['embedding.weight'].shape[0] if 'embedding.weight' in state_dict else len(vocab)

            model = MoleculeTransformer(
                vocab_size=vocab_size,
                d_model=params.get("embed", 256),
                nhead=params.get("nhead", 8),
                num_encoder_layers=params.get("layers", 4),
                num_decoder_layers=params.get("layers", 4),
                dim_feedforward=params.get("hidden", 1024),
                latent_size=128
            )
            model.load_state_dict(state_dict, strict=False)

        elif conf["type"] == "gru":
            vocab_size = state_dict['embedding.weight'].shape[0] if 'embedding.weight' in state_dict else len(vocab)
            
            model = MolecularVAE(
                vocab_size=vocab_size,
                embed_size=params.get("embed", 64),
                hidden_size=params.get("hidden", 256),
                latent_size=params.get("latent", 128),
                num_layers=params.get("layers", 3)
            )
            model.load_state_dict(state_dict, strict=False)
        
        else:
            raise ValueError(f"Unknown model type: {conf['type']}")

        model.to(Config.DEVICE)
        model.eval()
        logger.info(f"Model '{conf['name']}' loaded successfully.")
        return model

    except Exception as e:
        logger.critical(f"Failed to load {model_key}: {e}", exc_info=True)
        return None

def format_report(smiles, properties, target_info, affinity, similarity, recipe, naming_service):
    """Generates the MedChem scientific report."""
    
    # 1. Novelty Check
    is_novel, name, link = ChemistryService.check_novelty(smiles)
    if is_novel:
        header = "**NOVEL ENTITY (AI-Generated)**"
        scaffold_info = "Scaffold Novelty: **High**"
    else:
        header = f"**KNOWN COMPOUND:** [{name}]({link})"
        scaffold_info = "Scaffold Novelty: **Low** (Known)"

    # 2. Target Information
    target_block = ""
    if target_info:
        target_block = (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"**TARGET PROFILE**\n"
            f"Target: `{target_info['target_name']}`\n"
            f"Class: _{target_info['target_class']}_\n"
        )

    # 3. Biology / Docking
    bio_block = ""
    if target_info and affinity is not None:
        verdict = BiologyService.interpret_affinity(affinity)
        confidence = BiologyService.get_confidence_level(affinity, similarity)
        
        bio_block = (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"**DOCKING SIMULATION**\n"
            f"Score: `{affinity} kcal/mol`\n"
            f"Verdict: {verdict}\n"
            f"Confidence: `{confidence}`\n"
        )
    
    # 4. Chemistry & Risks
    iupac = naming_service.get_iupac_name(smiles) if naming_service else "N/A"
    
    chem_block = (
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 **MOLECULAR PROPERTIES**\n"
        f"MW: `{properties['mw']}` | TPSA: `{properties['tpsa']} Å²`\n"
        f"cLogP: `{properties['logp']}` | **cLogD7.4:** `{properties['clogd']}`\n"
        f"State: _{properties['pka_type']}_\n"
        f"CNS Prob: `{properties['cns_prob']}`\n"
        f"QED: `{properties['qed']}`\n\n"
        f"**RISK ASSESSMENT**\n"
        f"{properties['risk_profile']}\n"
        f"• {scaffold_info}\n"
    )

    # 5. Synthesis
    synth_block = (
        f"━━━━━━━━━━━━━━━━━━\n"
        f"**SYNTHESIS**\n"
        f"{recipe}\n"
    )

    return (
        f"{header}\n"
        f"Name: _{iupac}_\n"
        f"SMILES: `{smiles}`\n"
        f"{target_block}"
        f"{bio_block}"
        f"{chem_block}"
        f"{synth_block}"
    )

# --- MAIN BOT CONTROLLER ---

class MolecularBot:
    def __init__(self):
        logger.info("Initializing MUG Bot Controller...")
        self.bot = telebot.TeleBot(Config.API_TOKEN)
        
        # State Management
        self.current_model_key = Config.CURRENT_MODEL_KEY
        self.current_model_type = Config.MODEL_REGISTRY[self.current_model_key]["type"]
        
        # Load Core Components
        vocab_path = Config.MODEL_REGISTRY[self.current_model_key]["vocab"]
        self.vocab = load_vocabulary(vocab_path)
        self.model = load_model_instance(self.current_model_key, self.vocab)
        
        # Load Services
        self.tox_service = ToxicityService()
        try: 
            self.naming_service = LinguisticsService()
        except Exception: 
            self.naming_service = None
            logger.warning("Linguistics service unavailable.")
            
        self.session_cache = {}
        
        # Deep Search State: {chat_id: {active, best_cand, score, iterations, target}}
        self.deep_search_state = {} 
        
        self._register_handlers()
        logger.info("MUG System Ready.")

    def change_model(self, model_key: str) -> bool:
        """Hot-swap the active neural model."""
        if model_key not in Config.MODEL_REGISTRY:
            return False
            
        new_conf = Config.MODEL_REGISTRY[model_key]
        self.vocab = load_vocabulary(new_conf["vocab"])
        new_model = load_model_instance(model_key, self.vocab)
        
        if new_model:
            self.model = new_model
            self.current_model_key = model_key
            self.current_model_type = new_conf["type"]
            return True
        return False

    def process_generated_sequence(self, sequences, target_fp, target_cat, is_targeted) -> List[Tuple]:
        """
        Core logic: Decodes, Normalizes, Validates, Predicts Toxicity, Scores.
        Returns list of (score, candidate_tuple).
        """
        candidates = []
        
        for seq in sequences:
            try:
                # 1. Decode
                text = self.vocab.decode(seq)
                smiles = sf.decoder(text)
                if not smiles: 
                    continue
                
                raw_mol = Chem.MolFromSmiles(smiles)
                if not raw_mol: 
                    continue

                # 2. Normalize (MedChem Rules)
                mol = ChemistryService.normalize_structure(raw_mol)
                smiles = Chem.MolToSmiles(mol)
                
                if Descriptors.MolWt(mol) < 100: 
                    continue
                
                # 3. Analyze Properties
                props = ChemistryService.analyze_properties(mol)
                
                # 4. CNS Validation (Rules)
                cns_penalty, cns_warnings = ChemistryService.validate_cns_rules(mol, target_cat)
                
                # 5. AI Toxicity Prediction (ML Models)
                ai_risks = self.tox_service.predict(mol)
                
                # Merge Risks
                full_risks = []
                if cns_warnings: 
                    full_risks.extend([f"{w}" for w in cns_warnings])
                if ai_risks: 
                    full_risks.extend(ai_risks)
                
                props['risk_profile'] = "\n".join(full_risks) if full_risks else "No AI-predicted risks"

                # 6. Scoring
                score = -1000
                affinity = 0.0
                sim = 0.0

                if not is_targeted:
                    # Random Mode: Just check basic validity & safety
                    if props['qed'] > 0.1 and cns_penalty == 0:
                        score = props['qed']
                        candidates.append((score, (mol, smiles, props, None, None)))
                else:
                    # Targeted Mode: Full Evaluation
                    sim = DataStructs.TanimotoSimilarity(target_fp, AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
                    
                    # Run Docking only for promising structures to save CPU
                    if props['qed'] > 0.25:
                        affinity = BiologyService().dock_molecule(mol, target_cat)
                    
                    # MUG Scoring Function v2
                    score = (sim * 50) + (props['qed'] * 30) + abs(affinity) * 5
                    
                    # Penalties
                    score -= cns_penalty
                    
                    # Toxicity Penalties
                    if any("High" in r for r in ai_risks): 
                        score -= 40
                    elif any("Medium" in r for r in ai_risks): 
                        score -= 15
                    
                    candidates.append((score, (mol, smiles, props, affinity, sim)))
                    
            except Exception:
                continue
            
        return candidates

    def run_deep_search_worker(self, chat_id, target_info, target_cat):
        """Background worker loop for Deep Search."""
        logger.info(f"Deep Search started for {chat_id}")
        
        target_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(target_info['ref']), 2, nBits=1024)
        batch_size = 50
        
        while self.deep_search_state.get(chat_id, {}).get('active', False):
            # Generate
            with torch.no_grad():
                try:
                    if self.current_model_type == "transformer":
                        indices = self.model.sample(batch_size, Config.DEVICE, self.vocab, max_len=150, temperature=0.7)
                    else:
                        indices = self.model.sample(batch_size, Config.DEVICE, self.vocab, max_len=120, temp=0.7)
                except Exception as e:
                    logger.error(f"Deep Search Gen Error: {e}")
                    time.sleep(1)
                    continue

            # Process
            candidates = self.process_generated_sequence(indices.cpu().numpy(), target_fp, target_cat, is_targeted=True)
            
            # Update Best
            state = self.deep_search_state[chat_id]
            state['iterations'] += batch_size
            
            for score, cand_data in candidates:
                if state['best_candidate'] is None or score > state['best_score']:
                    state['best_score'] = score
                    state['best_candidate'] = cand_data
                    
            time.sleep(0.1)

    # --- HANDLERS ---

    def _register_handlers(self):
        
        @self.bot.message_handler(commands=['start', 'help'])
        def handle_start(message):
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            markup.row("Random Synthesis", "Targeted Design")
            markup.row("Change Model")
            text = (
                f"**MUG System v7.4**\n"
                f"Core: `{Config.MODEL_REGISTRY[self.current_model_key]['name']}`\n"
                f"Toxicity Engine: Active (12 models)\n\n"
                "Select operation mode:"
            )
            self.bot.send_message(message.chat.id, text, parse_mode='Markdown', reply_markup=markup)

        # Model Switching
        @self.bot.message_handler(func=lambda m: m.text == "Change Model")
        def handle_model_menu(message):
            markup = types.InlineKeyboardMarkup(row_width=1)
            for key, conf in Config.MODEL_REGISTRY.items():
                markup.add(types.InlineKeyboardButton(f"{conf['name']}", callback_data=f"mdl_{key}"))
            self.bot.send_message(message.chat.id, "**Select Architecture:**", parse_mode='Markdown', reply_markup=markup)

        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("mdl_"))
        def callback_model_switch(call):
            key = call.data.split("mdl_")[1]
            if key == self.current_model_key:
                self.bot.answer_callback_query(call.id, "Already active!")
                return
            
            self.bot.answer_callback_query(call.id, "Loading weights...")
            if self.change_model(key):
                self.bot.edit_message_text(f"Loaded: `{Config.MODEL_REGISTRY[key]['name']}`", call.message.chat.id, call.message.message_id, parse_mode='Markdown')
            else:
                self.bot.edit_message_text("Failed to load model.", call.message.chat.id, call.message.message_id)

        # Standard Modes
        @self.bot.message_handler(func=lambda m: m.text == "Random Synthesis")
        def handle_random(message):
            self.execute_standard_pipeline(message.chat.id, mode="random")

        @self.bot.message_handler(func=lambda m: m.text == "Targeted Design")
        def handle_targeted(message):
            markup = types.InlineKeyboardMarkup(row_width=2)
            for key, value in Config.DISEASE_DB.items():
                markup.add(types.InlineKeyboardButton(value['title'], callback_data=f"cat_{key}"))
            self.bot.send_message(message.chat.id, "**Select Therapeutic Area:**", parse_mode='Markdown', reply_markup=markup)

        # Category Selection
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("cat_"))
        def callback_category(call):
            cat = call.data.split("_")[1]
            markup = types.InlineKeyboardMarkup(row_width=2)
            for key, val in Config.DISEASE_DB[cat]['targets'].items():
                markup.add(types.InlineKeyboardButton(val['target_name'], callback_data=f"tgt_{cat}_{key}"))
            self.bot.edit_message_text(f"**{Config.DISEASE_DB[cat]['title']}**", call.message.chat.id, call.message.message_id, reply_markup=markup, parse_mode='Markdown')

        # Target Selection -> Search Mode
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("tgt_"))
        def callback_target(call):
            _, cat, dis = call.data.split("_")
            target = Config.DISEASE_DB[cat]['targets'][dis]
            
            markup = types.InlineKeyboardMarkup(row_width=1)
            markup.add(
                types.InlineKeyboardButton("Normal Search (Fast)", callback_data=f"run_norm_{cat}_{dis}"),
                types.InlineKeyboardButton("Deep Search (Infinite)", callback_data=f"run_deep_{cat}_{dis}")
            )
            
            self.bot.edit_message_text(
                f"**Target Locked:** {target['target_name']}\n"
                f"Choose strategy:", 
                call.message.chat.id, call.message.message_id, parse_mode='Markdown', reply_markup=markup
            )

        # Execute Search
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("run_"))
        def callback_run(call):
            _, mode, cat, dis = call.data.split("_")
            target = Config.DISEASE_DB[cat]['targets'][dis]
            
            if mode == "norm":
                self.bot.edit_message_text("Running Standard Inference...", call.message.chat.id, call.message.message_id)
                self.execute_standard_pipeline(call.message.chat.id, "targeted", target, cat)
            
            elif mode == "deep":
                # Initialize State with Target Info
                self.deep_search_state[call.message.chat.id] = {
                    'active': True,
                    'best_candidate': None,
                    'best_score': -1000,
                    'iterations': 0,
                    'target_info': target # Save target info for later
                }
                
                # Start Thread
                t = threading.Thread(target=self.run_deep_search_worker, args=(call.message.chat.id, target, cat))
                t.daemon = True
                t.start()
                
                markup = types.InlineKeyboardMarkup()
                markup.add(types.InlineKeyboardButton("STOP & SHOW RESULT", callback_data="stop_deep"))
                
                self.bot.edit_message_text(
                    f"**DEEP SEARCH ACTIVE**\n"
                    f"Target: `{target['target_name']}`\n\n"
                    f"AI is exploring chemical space indefinitely.\n"
                    f"Press STOP to retrieve the best candidate.",
                    call.message.chat.id, call.message.message_id, parse_mode='Markdown', reply_markup=markup
                )

        # Stop Deep Search
        @self.bot.callback_query_handler(func=lambda call: call.data == "stop_deep")
        def callback_stop_deep(call):
            cid = call.message.chat.id
            if cid in self.deep_search_state and self.deep_search_state[cid]['active']:
                self.deep_search_state[cid]['active'] = False
                
                self.bot.edit_message_text("Stopping... Analyzing results...", cid, call.message.message_id)
                time.sleep(1.5) # Allow thread to finish
                
                state = self.deep_search_state[cid]
                cand = state['best_candidate']
                saved_target = state.get('target_info')
                
                if cand and saved_target:
                    self.send_final_report(
                        cid, 
                        cand, 
                        saved_target, 
                        extra_text=f"**Deep Search Finished**\nScanned: {state['iterations']} molecules"
                    )
                else:
                    self.bot.send_message(cid, "Search stopped. No valid candidates found.")
            else:
                self.bot.answer_callback_query(call.id, "No active search.")

        # Actions (3D, Blocks, Rerun, PDF)
        @self.bot.callback_query_handler(func=lambda call: call.data in ["get_recipe", "get_3d", "refresh_random", "get_pdf"])
        def handle_actions(call):
            cid = call.message.chat.id
            smi = self.session_cache.get(cid)
            
            if not smi:
                self.bot.answer_callback_query(call.id, "Session expired. Please generate a new molecule.")
                return

            mol = Chem.MolFromSmiles(smi)
            if not mol:
                self.bot.answer_callback_query(call.id, "Error parsing molecule.")
                return

            if call.data == "get_3d":
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
                sio = io.StringIO()
                w = Chem.SDWriter(sio)
                w.write(mol_3d)
                w.close()
                bio = io.BytesIO(sio.getvalue().encode('utf-8'))
                bio.name = 'structure.sdf'
                self.bot.send_document(cid, bio, caption="**3D Structure (SDF)**", parse_mode='Markdown')

            elif call.data == "get_recipe":
                blocks = RetrosynthesisService.get_building_blocks(mol)
                if blocks:
                    mols = [Chem.MolFromSmiles(b) for b in blocks]
                    img = Draw.MolsToGridImage(mols, molsPerRow=min(4, len(blocks)), subImgSize=(200, 200))
                    bio = io.BytesIO()
                    img.save(bio, format='PNG')
                    bio.seek(0)
                    self.bot.send_photo(cid, bio, caption="**Building Blocks (Retrosynthesis)**")
                else:
                    self.bot.answer_callback_query(call.id, "No simple precursors found.")

            elif call.data == "refresh_random":
                self.execute_standard_pipeline(cid, mode="random")

            elif call.data == "get_pdf":
                self.bot.send_chat_action(cid, 'upload_document')
                
                # Analyse again
                props = ChemistryService.analyze_properties(mol)
                aff = BiologyService().dock_molecule(mol, "unknown")
                
                # Save image for PDF
                Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                img_path = Config.CACHE_DIR / f"{cid}_temp.png"
                VisualizationService.draw_cyberpunk(mol).save(img_path)
                
                # Generate Text
                blocks = RetrosynthesisService.get_building_blocks(mol)
                complexity = RetrosynthesisService.assess_complexity(mol)
                recipe = RetrosynthesisService.describe_synthesis(blocks, complexity)
                
                # Create PDF
                pdf_filename = Config.CACHE_DIR / f"MUG_Report_{cid}.pdf"
                
                try:
                    ReportingService.generate_pdf(
                        mol_name=f"Candidate-{cid}",
                        smiles=smi,
                        props=props,
                        affinity=aff,
                        image_path=str(img_path),
                        recipe=recipe,
                        filename=str(pdf_filename)
                    )
                    
                    with open(pdf_filename, 'rb') as f:
                        self.bot.send_document(cid, f, caption="**Official Research Report**")
                        
                except Exception as e:
                    logger.error(f"PDF Error: {e}")
                    self.bot.send_message(cid, "Failed to generate PDF report.")

    def execute_standard_pipeline(self, chat_id, mode="random", target_info=None, target_cat=""):
        self.bot.send_chat_action(chat_id, 'upload_photo')
        
        target_fp = None
        if target_info:
            target_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(target_info['ref']), 2, nBits=1024)

        best_cand = None
        best_score = -1000
        attempts = 5 if mode == "targeted" else 10
        batch = 50 if mode == "targeted" else 10
        
        for _ in range(attempts):
            with torch.no_grad():
                try:
                    if self.current_model_type == "transformer":
                        idx = self.model.sample(batch, Config.DEVICE, self.vocab, max_len=150, temperature=0.7)
                    else:
                        idx = self.model.sample(batch, Config.DEVICE, self.vocab, max_len=120, temp=0.7)
                except Exception: 
                    continue
            
            cands = self.process_generated_sequence(idx.cpu().numpy(), target_fp, target_cat, mode=="targeted")
            
            for score, c in cands:
                if mode == "random":
                    best_cand = c
                    break
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand and mode == "random": 
                break
            
        if best_cand:
            self.send_final_report(chat_id, best_cand, target_info)
        else:
            self.bot.send_message(chat_id, "No viable candidates found.")

    def send_final_report(self, chat_id, candidate, target_info, extra_text=""):
        mol, smi, props, aff, sim = candidate
        self.session_cache[chat_id] = smi
        
        # Draw
        img = VisualizationService.draw_cyberpunk(mol)
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        bio.seek(0)
        
        # Synthesis Logic
        complex_lvl = RetrosynthesisService.assess_complexity(mol)
        precursors = RetrosynthesisService.get_building_blocks(mol)
        recipe = RetrosynthesisService.describe_synthesis(precursors, complex_lvl)
        
        # Format Report
        report = format_report(smi, props, target_info, aff, sim, recipe, self.naming_service)
        if extra_text: 
            report = f"{extra_text}\n\n{report}"
        
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("Blocks", callback_data="get_recipe"), 
            types.InlineKeyboardButton("3D", callback_data="get_3d"),
            types.InlineKeyboardButton("PDF Report", callback_data="get_pdf")
        )
        if not target_info: 
            markup.add(types.InlineKeyboardButton("Rerun", callback_data="refresh_random"))
        
        self.bot.send_photo(chat_id, bio, caption=report, parse_mode='Markdown', reply_markup=markup)

    def run(self):
        self.bot.polling(none_stop=True)

if __name__ == "__main__":
    app = MolecularBot()
    app.run()
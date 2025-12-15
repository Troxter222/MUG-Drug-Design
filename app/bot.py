"""
Molecular Universe Generator (MUG) - AI-Powered De Novo Drug Design Platform
Author: Ali (Troxter222)
License: MIT
Date: 2025

A deep learning-based system for generating novel molecular structures
with therapeutic potential using transformer-based VAE architecture.
"""

import io
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Tuple, Any

import selfies as sf
import telebot
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, Draw
from telebot import types

# Imports from project structure
from app.config import Config
from app.core.engine import MolecularVAE
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.biology import BiologyService
from app.services.chemistry import ChemistryService
from app.services.linguistics import LinguisticsService
from app.services.retrosynthesis import RetrosynthesisService
from app.services.visualization import VisualizationService


# --- LOGGING SETUP ---
def setup_logging() -> logging.Logger:
    """Configure professional logging system."""
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
    )
    
    # File Handler (Rotated)
    file_handler = RotatingFileHandler(
        Config.LOG_FILE, 
        maxBytes=5*1024*1024, 
        backupCount=3, 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Root Logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [] # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noise
    logging.getLogger("rdkit").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    rdBase.DisableLog('rdApp.*')
    
    return logging.getLogger("MUG.Controller")

logger = setup_logging()


def initialize_bot() -> telebot.TeleBot:
    """Initialize Telegram bot interface."""
    try:
        bot_instance = telebot.TeleBot(Config.API_TOKEN)
        logger.info(f"Bot initialized successfully. Device: {Config.DEVICE}")
        return bot_instance
    except Exception as e:
        logger.critical(f"Failed to initialize bot: {e}")
        raise

# --- MODEL MANAGEMENT ---

def load_vocabulary() -> Vocabulary:
    """Load vocabulary from JSON file."""
    try:
        with open(Config.VOCAB_PATH, 'r') as f:
            chars = json.load(f)
        
        required_tokens = ['<pad>', '<sos>', '<eos>']
        if '<sos>' not in chars:
            chars = required_tokens + sorted(chars)
        
        logger.info(f"Vocabulary loaded: {len(chars)} tokens")
        return Vocabulary(chars)
    except Exception as e:
        logger.critical(f"Failed to load vocabulary: {e}")
        sys.exit(1)


def load_model(vocab: Vocabulary) -> MoleculeTransformer:
    """Load and initialize the Transformer model."""
    if not os.path.exists(Config.CHECKPOINT_PATH):
        logger.critical(f"Model checkpoint not found at: {Config.CHECKPOINT_PATH}")
        sys.exit(1)
    
    try:
        # Load weights
        state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'fc_z' in k:
                new_key = k.replace('fc_z', 'fc_latent_to_hidden')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        state_dict = new_state_dict
        
        # Check vocab compatibility
        if 'embedding.weight' in state_dict:
            saved_vocab_size = state_dict['embedding.weight'].shape[0]
        else:
            # Fallback если ключи совсем другие (маловероятно)
            saved_vocab_size = len(vocab)

        current_vocab_size = len(vocab)
        
        if saved_vocab_size != current_vocab_size:
            logger.warning(
                f"⚠️ Resizing model embeddings: {current_vocab_size} -> {saved_vocab_size}"
            )
            model_vocab_size = saved_vocab_size
        else:
            model_vocab_size = current_vocab_size
            
        # Initialize Architecture
        model = MoleculeTransformer(
            vocab_size=model_vocab_size,
            d_model=Config.EMBED_SIZE,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_LAYERS,
            num_decoder_layers=Config.NUM_LAYERS,
            dim_feedforward=Config.HIDDEN_SIZE,
            latent_size=Config.LATENT_SIZE
        ).to(Config.DEVICE)
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        logger.info(f"✅ Model loaded successfully on {Config.DEVICE}")
        return model
        
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        sys.exit(1)


def initialize_services():
    """Initialize external service components."""
    try:
        naming_service = LinguisticsService()
    except Exception as e:
        logger.warning(f"Linguistics service unavailable: {e}")
        naming_service = None
    
    return naming_service


def calculate_molecule_score(
    mol: Chem.Mol,
    properties: Dict,
    target_fp: Optional[DataStructs.ExplicitBitVect],
    target_category: str
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Evaluate molecule quality using multi-objective scoring.
    
    Args:
        mol: RDKit molecule object
        properties: Dictionary of computed molecular properties
        target_fp: Target fingerprint for similarity calculation
        target_category: Therapeutic category for docking
        
    Returns:
        Tuple of (score, affinity, similarity)
    """
    if target_fp is None:
        return properties['qed'], None, None
    
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    similarity = DataStructs.TanimotoSimilarity(target_fp, fingerprint)
    
    affinity = 0.0
    if properties['qed'] > 0.2:
        affinity = BiologyService.dock_simulation(mol, target_category)
    
    score = (similarity * 50) + (properties['qed'] * 30) + abs(affinity) * 5
    
    if "⚠️" in properties['toxicity']:
        score -= 50
    
    return score, affinity, similarity


def generate_molecules(
    model: MoleculeTransformer,
    vocab: Vocabulary,
    target_info: Optional[Dict],
    target_category: str
) -> Optional[Tuple]:
    """
    Core generation pipeline with adaptive sampling strategy.
    
    Args:
        model: Transformer model for generation
        vocab: Vocabulary for decoding
        target_info: Target specification for guided generation
        target_category: Therapeutic category
        
    Returns:
        Best candidate as tuple (mol, smiles, properties, affinity, similarity)
    """
    is_targeted = target_info is not None
    batch_size = 50 if is_targeted else 10
    max_attempts = 5 if is_targeted else 10
    
    target_fp = None
    if is_targeted:
        reference_mol = Chem.MolFromSmiles(target_info['ref'])
        target_fp = AllChem.GetMorganFingerprintAsBitVect(reference_mol, 2, nBits=1024)
    
    best_candidate = None
    best_score = -1000
    
    for attempt in range(max_attempts):
        with torch.no_grad():
            indices = model.sample(
                batch_size, 
                Config.DEVICE, 
                vocab, 
                max_len=200, 
                temperature=1.0
            )
        
        sequences = indices.cpu().numpy()
        
        for seq in sequences:
            try:
                smiles = sf.decoder(vocab.decode(seq))
                if not smiles:
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    continue
                
                Chem.SanitizeMol(mol)
                
                if Descriptors.MolWt(mol) < 100:
                    continue
                
                properties = ChemistryService.analyze_properties(mol)
                
                if not is_targeted:
                    if properties['qed'] > 0.1:
                        return (mol, smiles, properties, None, None)
                else:
                    score, affinity, similarity = calculate_molecule_score(
                        mol, properties, target_fp, target_category
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = (mol, smiles, properties, affinity, similarity)
                        
            except Exception:
                continue
        
        if best_candidate and not is_targeted:
            break
    
    return best_candidate


def format_report(
    smiles: str,
    properties: Dict,
    target_info: Optional[Dict],
    affinity: Optional[float],
    similarity: Optional[float],
    recipe: str,
    naming_service: Optional[LinguisticsService]
) -> str:
    """Generate formatted molecule report."""
    is_novel, name, link = ChemistryService.check_novelty(smiles)
    
    iupac_name = "Synthetic Compound"
    if naming_service:
        try:
            iupac_name = naming_service.get_iupac_name(smiles)
        except Exception:
            pass
    
    if is_novel:
        header = "✨ **NOVEL ENTITY**"
        name_line = f"🏷 **Name (AI):** _{iupac_name}_"
    else:
        header = f"🌍 **Known:** [{name}]({link})"
        name_line = f"🏷 **Name:** _{name}_"
    
    mode_line = (
        f"🎯 **Objective:** {target_info['name']}\n" 
        if target_info 
        else "👽 **Mode:** Random Exploration\n"
    )
    
    bio_section = ""
    if target_info and affinity is not None:
        verdict = BiologyService.interpret_affinity(affinity)
        bio_section = (
            f"\n🧬 **BIO-PHYSICS:**\n"
            f"🔗 Affinity: `{affinity} kcal/mol`\n"
            f"🎯 Similarity: `{similarity*100:.1f}%`\n"
            f"🏷 Verdict: {verdict}\n"
        )
    
    return (
        f"{header}\n"
        f"{mode_line}"
        f"{name_line}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🧬 **SMILES:** `{smiles}`\n"
        f"{bio_section}\n"
        f"🏗 **SYNTHESIS ROUTE:**\n{recipe}\n\n"
        f"📊 **MOLECULAR PROFILE:**\n"
        f"💊 QED: `{properties['qed']}` | ⚖️ MW: `{properties['mw']}`\n"
        f"🧠 BBB: {properties['brain']} | ☠️ Tox: {properties['toxicity']}"
    )


def execute_generation_pipeline(
    chat_id: int,
    bot: telebot.TeleBot,
    model: MoleculeTransformer,
    vocab: Vocabulary,
    naming_service: Optional[LinguisticsService],
    session_cache: Dict[int, str],
    target_info: Optional[Dict] = None,
    target_category: str = ""
) -> None:
    """Execute complete molecule generation and reporting pipeline."""
    bot.send_chat_action(chat_id, 'upload_photo')
    
    candidate = generate_molecules(model, vocab, target_info, target_category)
    
    if not candidate:
        bot.send_message(chat_id, "⚠️ No viable candidates found. Please try again.")
        return
    
    mol, smiles, properties, affinity, similarity = candidate
    session_cache[chat_id] = smiles
    
    image = VisualizationService.draw_cyberpunk(mol)
    image_buffer = io.BytesIO()
    image.save(image_buffer, format='PNG')
    image_buffer.seek(0)
    
    precursors = RetrosynthesisService.get_building_blocks(mol)
    synthesis_recipe = RetrosynthesisService.describe_synthesis(precursors)
    
    caption = format_report(
        smiles, properties, target_info, affinity, 
        similarity, synthesis_recipe, naming_service
    )
    
    keyboard = types.InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        types.InlineKeyboardButton("🧱 Precursors", callback_data="get_recipe"),
        types.InlineKeyboardButton("📦 3D Model", callback_data="get_3d"),
        types.InlineKeyboardButton(
            "🔄 Rerun", 
            callback_data="refresh_target" if target_info else "refresh_random"
        )
    )
    
    bot.send_photo(
        chat_id, 
        image_buffer, 
        caption=caption, 
        parse_mode='Markdown', 
        reply_markup=keyboard
    )

# --- MAIN BOT CLASS ---

class MolecularBot:
    """
    Main controller for the Telegram Bot.
    Encapsulates state, model, and handlers.
    """
    
    def __init__(self):
        logger.info("Initializing MUG Bot...")
        
        self.bot = telebot.TeleBot(Config.API_TOKEN)
        self.vocab = load_vocabulary()
        self.model = load_model(self.vocab)
        
        # Initialize Services
        try:
            self.naming_service = LinguisticsService()
        except Exception:
            self.naming_service = None
            logger.warning("Linguistics service disabled.")
            
        # User Session Memory (Chat ID -> Last SMILES)
        self.session_cache: Dict[int, str] = {}
        
        # Register Handlers
        self._register_handlers()
        logger.info("🤖 Bot Initialization Complete.")

    def _register_handlers(self):
        """Bind Telegram events to methods."""
        
        @self.bot.message_handler(commands=['start', 'help'])
        def handle_start(message):
            self.send_welcome(message)
            
        @self.bot.message_handler(commands=['status'])
        def handle_status(message):
            self.send_status(message)
            
        @self.bot.message_handler(func=lambda m: m.text == "🧬 Random Synthesis")
        def handle_random(message):
            self.execute_pipeline(message.chat.id, mode="random")
            
        @self.bot.message_handler(func=lambda m: m.text == "🎯 Targeted Design")
        def handle_targeted(message):
            self.send_target_categories(message)
            
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("cat_"))
        def callback_category(call):
            self.send_target_diseases(call)
            
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("tgt_"))
        def callback_target(call):
            self.run_hunter_protocol(call)

        @self.bot.callback_query_handler(func=lambda call: True)
        def callback_actions(call):
            self.handle_actions(call)

    # --- UI METHODS ---

    def send_welcome(self, message):
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        markup.add(
            types.KeyboardButton("🧬 Random Synthesis"),
            types.KeyboardButton("🎯 Targeted Design")
        )
        text = (
            "🌌 **MUG System v7.2 (Enterprise)**\n"
            "AI-Driven De Novo Drug Design Platform\n\n"
            "Select operating mode:"
        )
        self.bot.send_message(message.chat.id, text, parse_mode='Markdown', reply_markup=markup)

    def send_status(self, message):
        """Admin command to check system health."""
        mem_usage = "N/A"
        if torch.cuda.is_available():
            mem_usage = f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
            
        text = (
            f"🖥 **System Status**\n"
            f"Device: `{Config.DEVICE}`\n"
            f"GPU Memory: `{mem_usage}`\n"
            f"Model: Transformer V2\n"
            f"Vocab Size: {len(self.vocab)}"
        )
        self.bot.send_message(message.chat.id, text, parse_mode='Markdown')

    def send_target_categories(self, message):
        markup = types.InlineKeyboardMarkup(row_width=2)
        for key, value in Config.DISEASE_DB.items():
            markup.add(types.InlineKeyboardButton(value['title'], callback_data=f"cat_{key}"))
        self.bot.send_message(message.chat.id, "🔬 **Select Therapeutic Area:**", parse_mode='Markdown', reply_markup=markup)

    def send_target_diseases(self, call):
        category = call.data.split("_")[1]
        markup = types.InlineKeyboardMarkup(row_width=2)
        for key, value in Config.DISEASE_DB[category]['targets'].items():
            markup.add(types.InlineKeyboardButton(value['name'], callback_data=f"tgt_{category}_{key}"))
        markup.add(types.InlineKeyboardButton("🔙 Back", callback_data="back_home"))
        
        self.bot.edit_message_text(
            f"📂 **{Config.DISEASE_DB[category]['title']}**\nSelect pathology:",
            call.message.chat.id, call.message.message_id, reply_markup=markup, parse_mode='Markdown'
        )

    # --- CORE LOGIC ---

    def calculate_score(self, mol, props, target_fp, target_category):
        """Scoring function for Targeted Mode."""
        if target_fp is None: 
            return 0
        
        # 1. Similarity
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        sim = DataStructs.TanimotoSimilarity(target_fp, fp)
        
        # 2. Docking (Only for promising candidates)
        affinity = 0.0
        if props['qed'] > 0.4:
            affinity = BiologyService.dock_simulation(mol, target_category)
            
        # 3. Weighted Score
        score = (sim * 50) + (props['qed'] * 30) + (abs(affinity) * 5)
        
        # Penalties
        if "⚠️" in props['toxicity']: 
            score -= 50
        
        return score, affinity, sim

    def execute_pipeline(self, chat_id, mode="random", target_info=None, target_cat=""):
        """Main Generation Pipeline."""
        self.bot.send_chat_action(chat_id, 'upload_photo')
        
        # Setup strategy
        is_targeted = (mode == "targeted")
        batch_size = 50 if is_targeted else 10
        attempts = 5 if is_targeted else 10
        
        target_fp = None
        if is_targeted and target_info:
            ref_mol = Chem.MolFromSmiles(target_info['ref'])
            target_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)

        best_candidate = None
        best_score = -1000
        
        # --- GENERATION LOOP ---
        for _ in range(attempts):
            with torch.no_grad():
                indices = self.model.sample(batch_size, Config.DEVICE, self.vocab, max_len=200, temperature=0.8)
            
            cpu_indices = indices.cpu().numpy()
            
            for i in range(batch_size):
                try:
                    # Decode
                    selfies_str = self.vocab.decode(cpu_indices[i])
                    smi = sf.decoder(selfies_str)
                    if not smi: 
                        continue
                    
                    mol = Chem.MolFromSmiles(smi)
                    if not mol: 
                        continue
                    Chem.SanitizeMol(mol)
                    
                    # Filter Junk
                    if Descriptors.MolWt(mol) < 100: 
                        continue
                    
                    # Analyze
                    props = ChemistryService.analyze_properties(mol)
                    
                    if not is_targeted:
                        # Random Mode: Accept first good molecule
                        if props['qed'] > 0.1:
                            best_candidate = (mol, smi, props, None, None)
                            break
                    else:
                        # Targeted Mode: Optimization
                        score, aff, sim = self.calculate_score(mol, props, target_fp, target_cat)
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = (mol, smi, props, aff, sim)
                            
                except Exception: 
                    continue
            
            if best_candidate and not is_targeted: 
                break
            
        # --- REPORTING ---
        if best_candidate:
            self.send_report(chat_id, best_candidate, target_info)
        else:
            self.bot.send_message(chat_id, "⚠️ No viable candidates found. Try again.")

    def run_hunter_protocol(self, call):
        """Wrapper for Targeted Design."""
        chat_id = call.message.chat.id
        _, cat, dis = call.data.split("_")
        target = Config.DISEASE_DB[cat]['targets'][dis]
        
        self.bot.edit_message_text(
            f"📡 **Target Locked:** {target['name']}\n⚙️ Running Deep Search...",
            chat_id, call.message.message_id, parse_mode='Markdown'
        )
        self.execute_pipeline(chat_id, mode="targeted", target_info=target, target_cat=cat)

    def send_report(self, chat_id, candidate, target_info):
        mol, smi, props, affinity, sim = candidate
        self.session_cache[chat_id] = smi
        
        # Image
        img = VisualizationService.draw_cyberpunk(mol)
        bio_io = io.BytesIO()
        img.save(bio_io, format='PNG')
        bio_io.seek(0)
        
        # Info
        is_new, name, link = ChemistryService.check_novelty(smi)
        iupac = self.naming_service.get_iupac_name(smi) if self.naming_service else "Synthetic Compound"
        
        # Retro
        precursors = RetrosynthesisService.get_building_blocks(mol)
        recipe = RetrosynthesisService.describe_synthesis(precursors)
        
        # Text Construction
        status = "✨ **NOVEL ENTITY**" if is_new else f"🌍 **Known:** [{name}]({link})"
        dock_info = ""
        
        if target_info:
            verdict = BiologyService.interpret_affinity(affinity)
            dock_info = (f"\n🧬 **BIO-PHYSICS:**\n"
                         f"🔗 Affinity: `{affinity} kcal/mol`\n"
                         f"🎯 Similarity: `{sim*100:.1f}%`\n"
                         f"🏷 Verdict: {verdict}\n")
            mode_info = f"🎯 **Objective:** {target_info['name']}"
        else:
            mode_info = "👽 **Mode:** Random Exploration"

        caption = (
            f"{status}\n{mode_info}\n"
            f"🏷 **Name:** _{iupac}_\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🧬 `{smi}`\n"
            f"{dock_info}\n"
            f"🏗 **SYNTHESIS:**\n{recipe}\n\n"
            f"📊 **PROFILE:**\n"
            f"💊 QED: `{props['qed']}` | ⚖️ MW: `{props['mw']}`\n"
            f"🧠 BBB: {props['brain']} | ☠️ Tox: {props['toxicity']}"
        )

        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.add(
            types.InlineKeyboardButton("🧱 Blocks", callback_data="get_recipe"),
            types.InlineKeyboardButton("📦 3D (.sdf)", callback_data="get_3d"),
            types.InlineKeyboardButton("🔄 Rerun", callback_data="refresh_target" if target_info else "refresh_random")
        )
        
        self.bot.send_photo(chat_id, bio_io, caption=caption, parse_mode='Markdown', reply_markup=markup)

    def handle_actions(self, call):
        """Handle button clicks."""
        chat_id = call.message.chat.id
        data = call.data
        
        if data == "back_home":
            self.send_target_categories(call.message) # Reuse existing method logic
            
        elif "refresh" in data:
            try: 
                self.bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
            except Exception: 
                pass
            
            if data == "refresh_random":
                self.execute_pipeline(chat_id, mode="random")
            else:
                self.bot.send_message(chat_id, "🔄 Please re-select target from menu.")

        elif data == "get_3d":
            smi = self.session_cache.get(chat_id)
            if smi:
                self.bot.send_chat_action(chat_id, 'upload_document')
                mol = Chem.MolFromSmiles(smi)
                mol_3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
                AllChem.MMFFOptimizeMolecule(mol_3d)
                
                sio = io.StringIO()
                writer = Chem.SDWriter(sio)
                writer.write(mol_3d)
                writer.close()
                bio = io.BytesIO(sio.getvalue().encode('utf-8'))
                bio.name = 'structure_3d.sdf'
                self.bot.send_document(chat_id, bio, caption="🧬 **3D Model**")
                
        elif data == "get_recipe":
            smi = self.session_cache.get(chat_id)
            if smi:
                mol = Chem.MolFromSmiles(smi)
                blocks = RetrosynthesisService.get_building_blocks(mol)
                if blocks:
                    img = Draw.MolsToGridImage([Chem.MolFromSmiles(b) for b in blocks], molsPerRow=len(blocks), subImgSize=(300,300))
                    bio = io.BytesIO()
                    img.save(bio, format='PNG')
                    bio.seek(0)
                    self.bot.send_photo(chat_id, bio, caption="🧩 **Precursors**")
                else:
                    self.bot.answer_callback_query(call.id, "No blocks found")

    def run(self):
        logger.info("🚀 MUG System Started.")
        self.bot.polling(none_stop=True)

if __name__ == "__main__":
    app = MolecularBot()
    app.run()
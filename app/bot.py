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
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Tuple

import selfies as sf
import telebot
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, Draw
from telebot import types

from app.config import Config
from app.core.engine import MolecularVAE
from app.core.transformer_model import MoleculeTransformer
from app.core.vocab import Vocabulary
from app.services.biology import BiologyService
from app.services.chemistry import ChemistryService
from app.services.linguistics import LinguisticsService
from app.services.retrosynthesis import RetrosynthesisService
from app.services.visualization import VisualizationService


def setup_logging() -> logging.Logger:
    """Configure logging system with file rotation and console output."""
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
    )
    
    file_handler = RotatingFileHandler(
        Config.LOG_FILE, 
        maxBytes=5*1024*1024, 
        backupCount=3, 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
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


def load_vocabulary() -> Vocabulary:
    """Load or initialize molecular vocabulary."""
    with open(Config.VOCAB_PATH, 'r') as f:
        chars = json.load(f)
    
    required_tokens = ['<pad>', '<sos>', '<eos>']
    if '<sos>' not in chars:
        chars = required_tokens + sorted(chars)
    
    return Vocabulary(chars)


def load_model(vocab: Vocabulary) -> MoleculeTransformer:
    """
    Load pre-trained transformer model with checkpoint validation.
    
    Args:
        vocab: Vocabulary object containing token mappings
        
    Returns:
        Loaded and initialized transformer model
    """
    if not os.path.exists(Config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {Config.CHECKPOINT_PATH}")
    
    model = MoleculeTransformer(
        vocab_size=len(vocab),
        d_model=Config.EMBED_SIZE,
        nhead=Config.NHEAD,
        num_encoder_layers=Config.NUM_LAYERS,
        num_decoder_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.HIDDEN_SIZE,
        latent_size=Config.LATENT_SIZE
    ).to(Config.DEVICE)
    
    state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    
    saved_vocab_size = state_dict['embedding.weight'].shape[0]
    if saved_vocab_size != len(vocab):
        logger.warning(
            f"Vocabulary size mismatch. Expected: {len(vocab)}, "
            f"Found: {saved_vocab_size}. Reinitializing model."
        )
        model = MoleculeTransformer(
            vocab_size=saved_vocab_size,
            d_model=Config.EMBED_SIZE,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_LAYERS,
            num_decoder_layers=Config.NUM_LAYERS,
            dim_feedforward=Config.HIDDEN_SIZE,
            latent_size=Config.LATENT_SIZE
        ).to(Config.DEVICE)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info("Model loaded successfully.")
    return model


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
                temp=1.0
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


class MolecularBot:
    """Main bot controller class."""
    
    def __init__(self):
        self.bot = initialize_bot()
        self.vocab = load_vocabulary()
        self.model = load_model(self.vocab)
        self.naming_service = initialize_services()
        self.session_cache: Dict[int, str] = {}
        
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all bot command and callback handlers."""
        
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
            keyboard.add(
                types.KeyboardButton("🧬 Random Synthesis"),
                types.KeyboardButton("🎯 Targeted Design")
            )
            
            welcome_text = (
                "🌌 **MUG System v7.1**\n"
                "AI-Driven De Novo Drug Design Platform\n\n"
                "Select operating mode:"
            )
            self.bot.send_message(
                message.chat.id, 
                welcome_text, 
                parse_mode='Markdown', 
                reply_markup=keyboard
            )
        
        @self.bot.message_handler(func=lambda m: m.text == "🧬 Random Synthesis")
        def handle_random_synthesis(message):
            execute_generation_pipeline(
                message.chat.id, 
                self.bot, 
                self.model, 
                self.vocab, 
                self.naming_service, 
                self.session_cache
            )
        
        @self.bot.message_handler(func=lambda m: m.text == "🎯 Targeted Design")
        def handle_targeted_menu(message):
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            for key, value in Config.DISEASE_DB.items():
                keyboard.add(
                    types.InlineKeyboardButton(
                        value['title'], 
                        callback_data=f"cat_{key}"
                    )
                )
            self.bot.send_message(
                message.chat.id, 
                "🔬 **Select Therapeutic Area:**", 
                parse_mode='Markdown', 
                reply_markup=keyboard
            )
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("cat_"))
        def handle_category_selection(call):
            category = call.data.split("_")[1]
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            
            for key, value in Config.DISEASE_DB[category]['targets'].items():
                keyboard.add(
                    types.InlineKeyboardButton(
                        value['name'], 
                        callback_data=f"tgt_{category}_{key}"
                    )
                )
            keyboard.add(
                types.InlineKeyboardButton("🔙 Back", callback_data="back_home")
            )
            
            self.bot.edit_message_text(
                f"📂 **{Config.DISEASE_DB[category]['title']}**\n"
                f"Select specific pathology:",
                call.message.chat.id,
                call.message.message_id,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("tgt_"))
        def handle_target_selection(call):
            _, category, disease = call.data.split("_")
            target = Config.DISEASE_DB[category]['targets'][disease]
            
            self.bot.edit_message_text(
                f"📡 **Target Locked:** {target['name']}\n"
                f"⚙️ Initializing generation pipeline...",
                call.message.chat.id,
                call.message.message_id,
                parse_mode='Markdown'
            )
            
            execute_generation_pipeline(
                call.message.chat.id,
                self.bot,
                self.model,
                self.vocab,
                self.naming_service,
                self.session_cache,
                target_info=target,
                target_category=category
            )
        
        @self.bot.callback_query_handler(func=lambda call: call.data == "back_home")
        def handle_back(call):
            handle_targeted_menu(call.message)
        
        @self.bot.callback_query_handler(func=lambda call: call.data.startswith("refresh"))
        def handle_refresh(call):
            try:
                self.bot.edit_message_reply_markup(
                    call.message.chat.id,
                    call.message.message_id,
                    reply_markup=None
                )
            except Exception:
                pass
            
            if call.data == "refresh_random":
                execute_generation_pipeline(
                    call.message.chat.id,
                    self.bot,
                    self.model,
                    self.vocab,
                    self.naming_service,
                    self.session_cache
                )
            else:
                self.bot.send_message(
                    call.message.chat.id,
                    "🔄 Please re-select target from the menu."
                )
        
        @self.bot.callback_query_handler(func=lambda call: call.data == "get_3d")
        def handle_3d_export(call):
            smiles = self.session_cache.get(call.message.chat.id)
            if not smiles:
                return
            
            self.bot.send_chat_action(call.message.chat.id, 'upload_document')
            
            mol = Chem.MolFromSmiles(smiles)
            mol_3d = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol_3d)
            
            string_buffer = io.StringIO()
            writer = Chem.SDWriter(string_buffer)
            writer.write(mol_3d)
            writer.close()
            
            file_buffer = io.BytesIO(string_buffer.getvalue().encode('utf-8'))
            file_buffer.name = 'structure_3d.sdf'
            
            self.bot.send_document(
                call.message.chat.id,
                file_buffer,
                caption="🧬 **3D Structural Data (.sdf)**"
            )
        
        @self.bot.callback_query_handler(func=lambda call: call.data == "get_recipe")
        def handle_precursor_view(call):
            smiles = self.session_cache.get(call.message.chat.id)
            if not smiles:
                return
            
            self.bot.send_chat_action(call.message.chat.id, 'upload_photo')
            
            mol = Chem.MolFromSmiles(smiles)
            precursors = RetrosynthesisService.get_building_blocks(mol)
            
            if not precursors:
                self.bot.answer_callback_query(
                    call.id, 
                    "No specific precursors identified."
                )
                return
            
            precursor_mols = [Chem.MolFromSmiles(p) for p in precursors]
            grid_image = Draw.MolsToGridImage(
                precursor_mols,
                molsPerRow=len(precursors),
                subImgSize=(300, 300)
            )
            
            image_buffer = io.BytesIO()
            grid_image.save(image_buffer, format='PNG')
            image_buffer.seek(0)
            
            self.bot.send_photo(
                call.message.chat.id,
                image_buffer,
                caption="🧩 **Synthetic Precursors**"
            )
    
    def run(self):
        """Start bot polling loop."""
        logger.info("System ready. Starting bot...")
        self.bot.polling(none_stop=True)


def main():
    """Application entry point."""
    try:
        bot_controller = MolecularBot()
        bot_controller.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
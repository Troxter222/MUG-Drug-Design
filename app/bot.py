"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import telebot
import torch
import json
import os
import io
import logging
from typing import Optional, Tuple, Dict

# External Libraries
import selfies as sf
from telebot import types
from rdkit import Chem, DataStructs
from logging.handlers import RotatingFileHandler
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit import rdBase

# MUG Modules
from app.config import Config
from app.core.engine import MolecularVAE
from app.core.vocab import Vocabulary
from app.services.chemistry import ChemistryService
from app.services.biology import BiologyService
from app.services.retrosynthesis import RetrosynthesisService
from app.services.visualization import VisualizationService
from app.services.linguistics import LinguisticsService

# --- CONFIGURATION & LOGGING ---
Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s')

file_handler = RotatingFileHandler(Config.LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger("BotController")

logging.getLogger("rdkit").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
rdBase.DisableLog('rdApp.*')

# --- SYSTEM INITIALIZATION ---
try:
    bot = telebot.TeleBot(Config.API_TOKEN)
    logger.info(f"🤖 Bot Interface Initialized. Device: {Config.DEVICE}")
except Exception as e:
    logger.critical(f"Failed to initialize TeleBot: {e}")
    exit(1)

# NLP Init
try:
    naming = LinguisticsService()
except Exception:
    naming = None

# Load AI Core
def load_ai_core() -> Tuple[Optional[MolecularVAE], Optional[Vocabulary]]:
    logger.info("⏳ Loading Neural Core...")
    
    try:
        # 1. Load Vocabulary
        with open(Config.VOCAB_PATH, 'r') as f:
            chars = json.load(f)
        
        required_tokens = ['<pad>', '<sos>', '<eos>']
        if '<sos>' not in chars:
            chars = required_tokens + sorted(chars)
            
        vocab = Vocabulary(chars)
        
        # 2. Load Model
        if not os.path.exists(Config.CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {Config.CHECKPOINT_PATH}")
            
        state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
        
        saved_vocab_size = state_dict['embedding.weight'].shape[0]
        
        if saved_vocab_size != len(vocab):
            logger.warning(f"⚠️ Resizing model embeddings: {len(vocab)} -> {saved_vocab_size}")
            model = MolecularVAE(saved_vocab_size, Config.EMBED_SIZE, Config.HIDDEN_SIZE, Config.LATENT_SIZE, Config.NUM_LAYERS)
        else:
            model = MolecularVAE(len(vocab), Config.EMBED_SIZE, Config.HIDDEN_SIZE, Config.LATENT_SIZE, Config.NUM_LAYERS)
            
        model.load_state_dict(state_dict)
        model.to(Config.DEVICE)
        model.eval()
        
        logger.info("✅ MUG System Online.")
        return model, vocab
        
    except Exception as e:
        logger.error(f"❌ Core Initialization Failed: {e}")
        return None, None

model, vocab = load_ai_core()
user_session_cache: Dict[int, str] = {}

# --- PIPELINE CONTROLLER ---

def run_generation_pipeline(chat_id: int, target_info: Optional[Dict] = None, target_cat: str = "") -> None:
    """
    Orchestrates the full generation process.
    """
    bot.send_chat_action(chat_id, 'upload_photo')
    
    candidate = None
    best_score = -1000
    
    is_targeted = target_info is not None
    batch_size = 50 if is_targeted else 10
    attempts = 5 if is_targeted else 10
    
    target_fp = None
    if is_targeted:
        ref_mol = Chem.MolFromSmiles(target_info['ref'])
        target_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)

    # --- GENERATION LOOP ---
    for _ in range(attempts):
        with torch.no_grad():
            indices = model.sample(batch_size, Config.DEVICE, vocab, max_len=200, temp=0.8)
        cpu_indices = indices.cpu().numpy()
        
        for i in range(batch_size):
            try:
                # Decode & Validate
                smi = sf.decoder(vocab.decode(cpu_indices[i]))
                if not smi: 
                    continue
                mol = Chem.MolFromSmiles(smi)
                if not mol: 
                    continue
                Chem.SanitizeMol(mol)
                
                if Descriptors.MolWt(mol) < 100: 
                    continue

                # --- SCORING ---
                props = ChemistryService.analyze_properties(mol)
                
                if not is_targeted:
                    # Random Mode
                    if props['qed'] > 0.5:
                        candidate = (mol, smi, props, None, None)
                        break
                else:
                    # Targeted Mode
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                    
                    affinity = 0.0
                    if props['qed'] > 0.5:
                        affinity = BiologyService.dock_simulation(mol, target_cat)
                        
                    score = (sim * 50) + (props['qed'] * 30) + abs(affinity) * 5
                    
                    # Penalties
                    if "⚠️" in props['toxicity']: 
                        score -= 50
                    
                    if score > best_score:
                        best_score = score
                        candidate = (mol, smi, props, affinity, sim)
            
            except Exception:
                continue # Skip invalid molecules
                        
        if candidate and not is_targeted: 
            break 
    
    # --- REPORT GENERATION ---
    if candidate:
        mol, smi, props, affinity, sim = candidate
        user_session_cache[chat_id] = smi 
        
        # 1. Visualization
        pil_image = VisualizationService.draw_cyberpunk(mol)
        bio_io = io.BytesIO()
        pil_image.save(bio_io, format='PNG')
        bio_io.seek(0)
        
        # 2. Retrosynthesis
        precursors = RetrosynthesisService.get_building_blocks(mol)
        recipe = RetrosynthesisService.describe_synthesis(precursors)
        
        # 3. Novelty & Naming
        is_new, name, link = ChemistryService.check_novelty(smi)
        
        try:
            iupac_name = naming.get_iupac_name(smi) if naming else "Synthetic Compound"
        except Exception: 
            iupac_name = "Synthetic Compound"

        status_header = "✨ **NOVEL ENTITY**" if is_new else f"🌍 **Known:** [{name}]({link})"
        name_block = f"🏷 **Name (AI):** _{iupac_name}_" if is_new else f"🏷 **Name:** _{name}_"
        
        # 4. Construct Message
        dock_block = ""
        if is_targeted:
            verdict = BiologyService.interpret_affinity(affinity)
            dock_block = (f"\n🧬 **BIO-PHYSICS:**\n"
                          f"🔗 Affinity: `{affinity} kcal/mol`\n"
                          f"🎯 Similarity: `{sim*100:.1f}%`\n"
                          f"🏷 Verdict: {verdict}\n")
            
        context_block = f"🎯 **Objective:** {target_info['name']}\n" if is_targeted else "👽 **Mode:** Random Exploration\n"
        
        caption = (
            f"{status_header}\n"
            f"{context_block}"
            f"{name_block}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🧬 **SMILES:** `{smi}`\n"
            f"{dock_block}\n"
            f"🏗 **SYNTHESIS ROUTE:**\n{recipe}\n\n"
            f"📊 **MOLECULAR PROFILE:**\n"
            f"💊 QED: `{props['qed']}` | ⚖️ MW: `{props['mw']}`\n"
            f"🧠 BBB: {props['brain']} | ☠️ Tox: {props['toxicity']}"
        )
        
        markup = types.InlineKeyboardMarkup(row_width=2)
        markup.add(
            types.InlineKeyboardButton("🧱 Precursors", callback_data="get_recipe"),
            types.InlineKeyboardButton("📦 3D Model", callback_data="get_3d"),
            types.InlineKeyboardButton("🔄 Rerun", callback_data="refresh_target" if is_targeted else "refresh_random")
        )
        
        bot.send_photo(chat_id, bio_io, caption=caption, parse_mode='Markdown', reply_markup=markup)
    else:
        bot.send_message(chat_id, "⚠️ Search yielded no viable candidates. Try again.")

# --- TELEGRAM HANDLERS ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton("🧬 Random Synthesis"), types.KeyboardButton("🎯 Targeted Design"))
    
    welcome_text = (
        "🌌 **MUG System v7.1 (Enterprise)**\n"
        "AI-Driven De Novo Drug Design Platform.\n\n"
        "Select operating mode:"
    )
    bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown', reply_markup=markup)

@bot.message_handler(func=lambda m: m.text == "🧬 Random Synthesis")
def handler_random(message):
    run_generation_pipeline(message.chat.id)

@bot.message_handler(func=lambda m: m.text == "🎯 Targeted Design")
def handler_targeted_menu(message):
    markup = types.InlineKeyboardMarkup(row_width=2)
    for k, v in Config.DISEASE_DB.items():
        markup.add(types.InlineKeyboardButton(v['title'], callback_data=f"cat_{k}"))
    bot.send_message(message.chat.id, "🔬 **Select Therapeutic Area:**", parse_mode='Markdown', reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith("cat_"))
def handler_disease_submenu(call):
    cat = call.data.split("_")[1]
    markup = types.InlineKeyboardMarkup(row_width=2)
    for k, v in Config.DISEASE_DB[cat]['targets'].items():
        markup.add(types.InlineKeyboardButton(v['name'], callback_data=f"tgt_{cat}_{k}"))
    markup.add(types.InlineKeyboardButton("🔙 Back", callback_data="back_home"))
    
    bot.edit_message_text(
        f"📂 **{Config.DISEASE_DB[cat]['title']}**\nSelect specific pathology:",
        call.message.chat.id, call.message.message_id, reply_markup=markup, parse_mode='Markdown'
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith("tgt_"))
def handler_run_hunter(call):
    chat_id = call.message.chat.id
    _, cat, dis = call.data.split("_")
    target = Config.DISEASE_DB[cat]['targets'][dis]
    
    bot.edit_message_text(
        f"📡 **Target Locked:** {target['name']}\n⚙️ Initializing Deep Generative Search...",
        chat_id, call.message.message_id, parse_mode='Markdown'
    )
    run_generation_pipeline(chat_id, target_info=target, target_cat=cat)

@bot.callback_query_handler(func=lambda call: True)
def handler_actions(call):
    chat_id = call.message.chat.id
    data = call.data
    
    if data == "back_home":
        handler_targeted_menu(call.message)
        
    elif "refresh" in data:
        try: 
            bot.edit_message_reply_markup(chat_id, call.message.message_id, reply_markup=None)
        except Exception: 
            pass
        
        if data == "refresh_random":
            run_generation_pipeline(chat_id)
        elif data == "refresh_target":
            bot.send_message(chat_id, "🔄 Please re-select the target from the menu above.")

    elif data == "get_3d":
        smi = user_session_cache.get(chat_id)
        if smi:
            bot.send_chat_action(chat_id, 'upload_document')
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
            bot.send_document(chat_id, bio, caption="🧬 **3D Structural Data (.sdf)**")
            
    elif data == "get_recipe":
        smi = user_session_cache.get(chat_id)
        if smi:
            bot.send_chat_action(chat_id, 'upload_photo')
            mol = Chem.MolFromSmiles(smi)
            blocks = RetrosynthesisService.get_building_blocks(mol)
            if blocks:
                mols = [Chem.MolFromSmiles(b) for b in blocks]
                img = Draw.MolsToGridImage(mols, molsPerRow=len(blocks), subImgSize=(300,300))
                bio = io.BytesIO()
                img.save(bio, format='PNG')
                bio.seek(0)
                bot.send_photo(chat_id, bio, caption="🧩 **Synthetic Precursors**")
            else:
                bot.answer_callback_query(call.id, "No specific precursors identified.")

if __name__ == "__main__":
    logger.info("🚀 System Ready. Polling...")
    bot.polling(none_stop=True)
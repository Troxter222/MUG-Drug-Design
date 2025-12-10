"""
Author: Ali (Troxter222)
Project: MUG (Molecular Universe Generator)
Date: 2025
License: MIT
"""

import os
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Suppress TensorFlow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class LinguisticsService:
    """
    Service responsible for Neural Machine Translation of chemical structures.
    Converts SMILES strings into human-readable IUPAC names using the STOUT model.
    """

    def __init__(self):
        self._is_active: bool = False
        self._translate_func = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Attempts to load the STOUT transformer model and the JVM backend.
        Handles missing Java dependencies gracefully without crashing.
        """
        logger.info("🧠 Initializing Linguistics Module (STOUT)...")
        
        try:
            # 1. Check if library exists
            import importlib.util
            if importlib.util.find_spec("STOUT") is None:
                raise ImportError("STOUT library not installed")

            # 2. Try import (This might fail if Java is missing)
            from STOUT import translate_forward
            
            # 3. Warm-up call (This triggers the JVM/JPype check)
            # If Java is missing, it crashes here with OSError or RuntimeException
            _ = translate_forward("C")
            
            self._translate_func = translate_forward
            self._is_active = True
            logger.info("✅ IUPAC Translator is online.")
            
        except (OSError, ImportError, Exception) as e:
            # Catch ALL errors (Missing DLLs, JPype errors, Missing JARs)
            error_msg = str(e)
            
            # Make the log friendly
            if "jpype" in error_msg.lower() or "jvm" in error_msg.lower():
                logger.warning("⚠️ JAVA/JVM not found. IUPAC naming disabled (Running in fallback mode).")
            else:
                logger.warning(f"⚠️ Linguistics module disabled: {error_msg}")

    def get_iupac_name(self, smiles: str) -> str:
        """
        Translates a SMILES string into a standardized IUPAC name.
        """
        # Fail-safe: Return generic name if model is not loaded
        if not self._is_active:
            return "Novel Synthetic Compound"

        try:
            # Perform neural translation
            name = self._translate_func(smiles)
            
            # Formatting: "2-ACETOXY..." -> "2-acetoxy..."
            if name:
                name = name.lower()
                if len(name) > 1:
                    name = name[0].upper() + name[1:]
                return name
            return "Synthetic Compound"
            
        except Exception:
            # If the model fails on a specific molecule
            return "Complex Synthetic Structure"

# Usage Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = LinguisticsService()
    print(f"Test: {service.get_iupac_name('CC(=O)O')}")
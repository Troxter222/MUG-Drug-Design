"""
Molecular Universe Generator (MUG) - Application Entry Point
Author: Ali (Troxter222)
License: MIT

Main launcher for the MUG Telegram bot interface.
"""

import logging
import os
import signal
import sys
from pathlib import Path

# Add project root to Python path for proper imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from app.bot import MolecularBot


def setup_logger() -> logging.Logger:
    """Configure application logger."""
    from rdkit import rdBase
    
    rdBase.DisableLog('rdApp.*')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("MUG.Launcher")


def validate_environment() -> bool:
    """
    Verify required environment variables and dependencies.
    
    Returns:
        True if environment is valid, False otherwise
    """
    required_vars = ['TELEGRAM_TOKEN']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"ERROR: Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True


def setup_signal_handlers(bot_instance: MolecularBot, logger: logging.Logger):
    """Configure graceful shutdown on SIGINT/SIGTERM."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}. Initiating shutdown...")
        try:
            bot_instance.bot.stop_polling()
            logger.info("Bot stopped successfully.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main application entry point."""
    logger = setup_logger()
    
    logger.info("=" * 60)
    logger.info("Molecular Universe Generator (MUG) v7.1")
    logger.info("AI-Powered De Novo Drug Design Platform")
    logger.info("=" * 60)
    
    if not validate_environment():
        logger.critical("Environment validation failed. Exiting.")
        sys.exit(1)
    
    try:
        logger.info("Initializing bot controller...")
        bot_controller = MolecularBot()
        
        setup_signal_handlers(bot_controller, logger)
        
        logger.info("System ready. Starting polling loop...")
        logger.info("Press Ctrl+C to stop.")
        
        bot_controller.run()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
        
    except ImportError as e:
        logger.critical(f"Failed to import required modules: {e}")
        logger.critical("Ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
        
    except FileNotFoundError as e:
        logger.critical(f"Required file not found: {e}")
        logger.critical("Verify model checkpoints and configuration files are present.")
        sys.exit(1)
        
    except ConnectionError as e:
        logger.critical(f"Network connection error: {e}")
        logger.critical("Check internet connectivity and Telegram API accessibility.")
        sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        logger.critical("Application crashed. Check logs for details.")
        sys.exit(1)
        
    finally:
        logger.info("Molecular Universe Generator terminated.")


if __name__ == "__main__":
    main()
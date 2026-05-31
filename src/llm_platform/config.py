import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Load environment variables from .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Retrieve API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def setup_logging(level: int = logging.INFO):
    """Configures centralized logging for the entire platform."""
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Standard format for logs: Timestamp [Level] (Logger_Name): Message
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(name)s): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 1. Console Handler (Outputs to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler (Saves to logs/app.log)
    file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8")
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    logging.getLogger("config").info("Logging system initialized successfully.")

# Automatically trigger logging setup on import
setup_logging()
"""
Logger centralizzato con Loguru
"""
import sys
from pathlib import Path
from loguru import logger
from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_ROTATION, LOG_RETENTION

# Rimuovi handler default
logger.remove()

# Console handler con colori
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
)

# File handler con rotazione
logger.add(
    LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    rotation=LOG_ROTATION,
    retention=LOG_RETENTION,
    compression="zip",
)

# Export
__all__ = ["logger"]
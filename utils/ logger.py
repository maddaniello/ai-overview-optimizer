"""
Logger centralizzato con Loguru
"""
import sys
from loguru import logger as loguru_logger

# Rimuovi handler default
loguru_logger.remove()

# Console handler con colori
loguru_logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Export
logger = loguru_logger

__all__ = ["logger"]

"""
Simple logger without loguru
"""
import sys
from datetime import datetime


class SimpleLogger:
    """Logger semplice che usa print"""

    @staticmethod
    def _format_msg(level, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {level} | {msg}"

    @staticmethod
    def info(msg):
        print(SimpleLogger._format_msg("INFO", msg))

    @staticmethod
    def error(msg, exc_info=False):
        print(SimpleLogger._format_msg("ERROR", msg), file=sys.stderr)

    @staticmethod
    def success(msg):
        print(SimpleLogger._format_msg("SUCCESS", msg))

    @staticmethod
    def warning(msg):
        print(SimpleLogger._format_msg("WARNING", msg))

    @staticmethod
    def debug(msg):
        print(SimpleLogger._format_msg("DEBUG", msg))


# Singleton instance
logger = SimpleLogger()


def get_logger(name: str = None) -> SimpleLogger:
    """Funzione factory per compatibilità"""
    return logger


__all__ = ["logger", "get_logger", "SimpleLogger"]

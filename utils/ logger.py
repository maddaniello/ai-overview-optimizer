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
    def error(msg):
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

logger = SimpleLogger()

__all__ = ["logger"]

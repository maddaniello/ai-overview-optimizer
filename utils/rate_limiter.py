"""
Rate limiter per API calls
"""
import time
from collections import deque
from functools import wraps
from utils.logger import logger


class RateLimiter:
    """Rate limiter con sliding window"""
    
    def __init__(self, max_calls: int, period: int, name: str = "API"):
        self.max_calls = max_calls
        self.period = period
        self.name = name
        self.calls = deque()
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Rimuovi chiamate vecchie
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Check limite
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                logger.warning(
                    f"Rate limit {self.name}: attendo {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
                
                # Rimuovi chiamata più vecchia
                self.calls.popleft()
            
            # Esegui chiamata
            self.calls.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper


# Export
__all__ = ["RateLimiter"]
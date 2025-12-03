"""
Configurazione centrale per AI Overview Content Optimizer
Multi-user mode: credenziali inserite manualmente
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica .env solo per sviluppo locale (opzionale)
load_dotenv()

# ==================== DIRECTORY ====================
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"

# Crea directory se non esistono
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ==================== DEFAULT CONFIG ====================
# Questi valori NON sono secrets, sono solo configurazioni

# DataForSEO
DATAFORSEO_API_URL = "https://api.dataforseo.com/v3"
DATAFORSEO_RATE_LIMIT = 5  # chiamate per minuto

# Location codes (DataForSEO)
LOCATION_CODES = {
    "Italy": 2380,
    "United States": 2840,
    "United Kingdom": 2826,
    "Germany": 2276,
    "France": 2250,
    "Spain": 2724,
}

# Language codes
LANGUAGE_CODES = {
    "Italian": "it",
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
}

# ==================== OPENAI CONFIG ====================
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 4096

# ==================== JINA CONFIG ====================
JINA_MODEL = "jina-reranker-v2-base-multilingual"
JINA_API_URL = "https://api.jina.ai/v1/rerank"

# ==================== SCRAPING CONFIG ====================
SCRAPING_TIMEOUT = 30
SCRAPING_MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# ==================== ANALYSIS CONFIG ====================
MAX_SOURCES = 5
MIN_ENTITY_FREQUENCY = 2
MAX_ANSWER_LENGTH = 300

# ==================== BRANDING MOCA ====================
MOCA_COLORS = {
    "primary": "#E52217",
    "secondary": "#FFE7E6",
    "dark": "#191919",
    "gray": "#8A8A8A",
    "white": "#FFFFFF",
}

MOCA_LOGO_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1-192x192.png"

# ==================== RATE LIMITS ====================
RATE_LIMITS = {
    "dataforseo": {"calls": 5, "period": 60},
    "openai": {"calls": 50, "period": 60},
    "jina": {"calls": 100, "period": 60},
    "scraping": {"calls": 10, "period": 60},
}

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"

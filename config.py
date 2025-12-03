"""
Configurazione centrale per AI Overview Content Optimizer
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

# ==================== DIRECTORY ====================
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"

# Crea directory se non esistono
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ==================== API KEYS ====================
DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN", "")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Google Cloud (opzionale)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID", "")

# ==================== RERANKER CONFIG ====================
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "jina").lower()

# Jina config
JINA_MODEL = "jina-reranker-v2-base-multilingual"
JINA_API_URL = "https://api.jina.ai/v1/rerank"

# Google Vertex AI config
GOOGLE_RANKING_MODEL = "semantic-ranker-512@latest"

# ==================== OPENAI CONFIG ====================
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_CHAT_MODEL = "gpt-4o"
OPENAI_MAX_TOKENS = 4096

# ==================== DATAFORSEO CONFIG ====================
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
    "Canada": 2124,
    "Australia": 2036,
}

# Language codes
LANGUAGE_CODES = {
    "Italian": "it",
    "English": "en",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
}

# ==================== SCRAPING CONFIG ====================
SCRAPING_TIMEOUT = 30  # secondi
SCRAPING_MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# ==================== ANALYSIS CONFIG ====================
MAX_SOURCES = 5  # numero massimo fonti da analizzare
MIN_ENTITY_FREQUENCY = 2  # frequenza minima entità per considerarla rilevante
MAX_ANSWER_LENGTH = 300  # parole massime per risposta estratta

# ==================== STREAMLIT CONFIG ====================
PAGE_TITLE = "AI Overview Content Optimizer"
PAGE_ICON = "🔍"

# ==================== BRANDING MOCA ====================
MOCA_COLORS = {
    "primary": "#E52217",      # Rosso Moca
    "secondary": "#FFE7E6",    # Rosa chiaro
    "dark": "#191919",         # Nero
    "gray": "#8A8A8A",         # Grigio
    "white": "#FFFFFF",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
}

MOCA_LOGO_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1-192x192.png"
MOCA_FONT = "Figtree"

# ==================== RATE LIMITS ====================
RATE_LIMITS = {
    "dataforseo": {"calls": 5, "period": 60},      # 5/min
    "openai": {"calls": 50, "period": 60},         # 50/min
    "jina": {"calls": 100, "period": 60},          # 100/min (free tier)
    "scraping": {"calls": 10, "period": 60},       # 10/min
}

# ==================== LOGGING CONFIG ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"

# ==================== VALIDATION ====================
def validate_config():
    """Valida configurazione API keys"""
    errors = []
    
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        errors.append("❌ DataForSEO credentials mancanti")
    
    if not OPENAI_API_KEY:
        errors.append("❌ OpenAI API key mancante")
    
    if RERANKER_PROVIDER == "jina" and not JINA_API_KEY:
        errors.append("❌ Jina API key mancante")
    
    if RERANKER_PROVIDER == "google":
        if not GOOGLE_APPLICATION_CREDENTIALS:
            errors.append("❌ Google credentials file mancante")
        if not GOOGLE_PROJECT_ID:
            errors.append("❌ Google Project ID mancante")
    
    return errors

# ==================== EXPORT ====================
__all__ = [
    "DATAFORSEO_LOGIN",
    "DATAFORSEO_PASSWORD",
    "OPENAI_API_KEY",
    "JINA_API_KEY",
    "RERANKER_PROVIDER",
    "LOCATION_CODES",
    "LANGUAGE_CODES",
    "MOCA_COLORS",
    "validate_config",
]
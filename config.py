"""
Configurazione centrale per AI Overview Content Optimizer
Multi-user mode: credenziali inserite manualmente
Supporto multi-modello: OpenAI + Gemini
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

# ==================== AI MODELS CONFIG ====================
# OpenAI Models
OPENAI_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "openai",
        "max_tokens": 4096,
        "description": "Modello più potente, multimodale"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "provider": "openai",
        "max_tokens": 4096,
        "description": "Veloce ed economico"
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "provider": "openai",
        "max_tokens": 4096,
        "description": "Alta qualità, context lungo"
    },
    "o1-preview": {
        "name": "o1 Preview",
        "provider": "openai",
        "max_tokens": 32768,
        "description": "Ragionamento avanzato"
    },
    "o1-mini": {
        "name": "o1 Mini",
        "provider": "openai",
        "max_tokens": 65536,
        "description": "Ragionamento veloce"
    },
}

# Google Gemini Models
GEMINI_MODELS = {
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro",
        "provider": "gemini",
        "max_tokens": 8192,
        "description": "Modello Pro multimodale"
    },
    "gemini-1.5-flash": {
        "name": "Gemini 1.5 Flash",
        "provider": "gemini",
        "max_tokens": 8192,
        "description": "Veloce ed economico"
    },
    "gemini-2.0-flash-exp": {
        "name": "Gemini 2.0 Flash",
        "provider": "gemini",
        "max_tokens": 8192,
        "description": "Ultima versione sperimentale"
    },
}

# All models combined
ALL_MODELS = {**OPENAI_MODELS, **GEMINI_MODELS}

# Default models
DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

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
MAX_SOURCES = 10
MIN_ENTITY_FREQUENCY = 2
MAX_ANSWER_LENGTH = 300
DEFAULT_ITERATIONS = 3
MAX_ITERATIONS = 10
DEFAULT_SERP_RESULTS = 10
MAX_SERP_RESULTS = 50

# ==================== AGENT CONFIG ====================
AGENT_TYPES = {
    "orchestrator": "Coordina il workflow completo",
    "serp": "Recupera dati SERP e AI Overview",
    "scraper": "Estrae contenuti dalle pagine",
    "analyzer": "Analizza contenuti ed entità",
    "optimizer": "Ottimizza risposte iterativamente",
    "ranking": "Calcola e confronta ranking",
    "strategy": "Genera analisi strategica e piano contenuto"
}

# ==================== BRANDING MOCA ====================
MOCA_COLORS = {
    "primary": "#E52217",
    "secondary": "#FFE7E6",
    "dark": "#191919",
    "gray": "#8A8A8A",
    "white": "#FFFFFF",
    "success": "#10B981",
    "warning": "#F59E0B",
    "info": "#3B82F6",
}

MOCA_LOGO_URL = "https://mocainteractive.com/wp-content/uploads/2025/04/cropped-moca-instagram-icona-1-192x192.png"

# ==================== RATE LIMITS ====================
RATE_LIMITS = {
    "dataforseo": {"calls": 5, "period": 60},
    "openai": {"calls": 50, "period": 60},
    "gemini": {"calls": 60, "period": 60},
    "jina": {"calls": 100, "period": 60},
    "scraping": {"calls": 10, "period": 60},
}

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"

"""
Helper functions utilities
"""
import hashlib
import re
from typing import List, Dict, Any
from urllib.parse import urlparse

def clean_text(text: str) -> str:
    """Pulisce testo da caratteri speciali e whitespace multipli"""
    # Rimuovi caratteri speciali mantenendo punteggiatura base
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"àèéìòù]', '', text)
    # Normalizza whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def truncate_text(text: str, max_length: int = 300, suffix: str = "...") -> str:
    """Tronca testo a lunghezza massima"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix

def extract_domain(url: str) -> str:
    """Estrae dominio da URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path
    except:
        return url

def hash_text(text: str) -> str:
    """Genera hash MD5 di un testo per caching"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def calculate_improvement_percentage(before: float, after: float) -> float:
    """Calcola percentuale di miglioramento"""
    if before == 0:
        return 0.0
    return ((after - before) / before) * 100

def format_score(score: float) -> str:
    """Formatta score per visualizzazione"""
    return f"{score:.2f}"

def deduplicate_list(items: List[Any], key: str = None) -> List[Any]:
    """Rimuove duplicati da lista mantenendo ordine"""
    if not key:
        seen = set()
        return [x for x in items if not (x in seen or seen.add(x))]
    
    seen = set()
    result = []
    for item in items:
        k = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisione sicura che evita division by zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

def extract_numbers_from_text(text: str) -> List[int]:
    """Estrae tutti i numeri da un testo"""
    return [int(n) for n in re.findall(r'\d+', text)]

def count_words(text: str) -> int:
    """Conta parole in un testo"""
    return len(text.split())

def validate_url(url: str) -> bool:
    """Valida se una stringa è un URL valido"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

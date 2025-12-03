"""
DataForSEO API Client
"""
import requests
import base64
from typing import Dict, Any, Optional
from utils.logger import logger
from utils.rate_limiter import RateLimiter
from config import DATAFORSEO_API_URL, DATAFORSEO_RATE_LIMIT

class DataForSEOClient:
    """Client per DataForSEO API"""
    
    def __init__(self, login: str, password: str):
        """
        Inizializza client con credenziali utente
        
        Args:
            login: DataForSEO login
            password: DataForSEO password
        """
        self.api_url = DATAFORSEO_API_URL
        self.login = login
        self.password = password
        
        # Crea auth header
        credentials = f"{login}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        }
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_calls=DATAFORSEO_RATE_LIMIT,
            period=60,
            name="DataForSEO"
        )
        
        logger.info("DataForSEO client inizializzato")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    @RateLimiter(max_calls=5, period=60)  # 5 chiamate al minuto
    def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Effettua richiesta POST a DataForSEO API
        
        Args:
            endpoint: Endpoint API (es. "/v3/serp/google/organic/live/advanced")
            payload: Payload JSON della richiesta
            
        Returns:
            Response JSON
        """
        url = f"{self.base_url}{endpoint}"
        
        logger.info(f"Richiesta a DataForSEO: {endpoint}")
        logger.debug(f"Payload: {payload}")
        
        try:
            response = requests.post(
                url,
                json=[payload],  # DataForSEO accetta array di task
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            # DataForSEO restituisce array di task
            if data.get("tasks") and len(data["tasks"]) > 0:
                task = data["tasks"][0]
                
                if task.get("status_code") == 20000:
                    logger.info(f"✓ Richiesta completata con successo")
                    return task
                else:
                    error_msg = task.get("status_message", "Unknown error")
                    logger.error(f"✗ Errore DataForSEO: {error_msg}")
                    raise Exception(f"DataForSEO Error: {error_msg}")
            
            raise Exception("No tasks in response")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Errore connessione: {str(e)}")
            raise
    
    def get_serp_with_ai_overview(
        self,
        keyword: str,
        location: str = "Italy",
        language: str = "Italian",
        device: str = "desktop"
    ) -> Dict[str, Any]:
        """
        Recupera SERP completo con AI Overview espanso
        
        Args:
            keyword: Keyword da cercare
            location: Località (es. "Italy")
            language: Lingua (es. "Italian")
            device: Dispositivo ("desktop" o "mobile")
            
        Returns:
            Dict con dati SERP e AI Overview
        """
        location_code = LOCATION_CODES.get(location, 2380)  # Default: Italy
        language_code = LANGUAGE_CODES.get(language, "it")  # Default: Italian
        
        payload = {
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "device": device,
            "os": "windows" if device == "desktop" else "android",
            "depth": 100,  # Recupera fino a 100 risultati
            "expand_ai_overview": True,  # CRITICAL: espande contenuto AI Overview
        }
        
        logger.info(f"🔍 Ricerca SERP per: '{keyword}' | Location: {location} | Language: {language}")
        
        result = self._make_request("/v3/serp/google/organic/live/advanced", payload)
        
        # Estrai dati rilevanti
        serp_data = self._extract_serp_data(result, keyword)
        
        return serp_data
    
    def _extract_serp_data(self, task_result: Dict, keyword: str) -> Dict[str, Any]:
        """
        Estrae e struttura dati SERP dalla risposta DataForSEO
        
        Args:
            task_result: Risultato task DataForSEO
            keyword: Keyword cercata
            
        Returns:
            Dict strutturato con dati SERP
        """
        if not task_result.get("result"):
            logger.warning("Nessun risultato trovato")
            return self._empty_serp_result(keyword)
        
        result = task_result["result"][0]
        items = result.get("items", [])
        
        # Estrai AI Overview
        ai_overview = None
        ai_overview_sources = []
        
        for item in items:
            if item.get("type") == "ai_overview":
                ai_overview = self._parse_ai_overview(item)
                ai_overview_sources = ai_overview.get("sources", [])
                logger.info(f"✓ AI Overview trovato con {len(ai_overview_sources)} fonti")
                break
        
        # Estrai risultati organici
        organic_results = []
        for item in items:
            if item.get("type") == "organic":
                organic_results.append({
                    "position": item.get("rank_absolute", 0),
                    "url": item.get("url", ""),
                    "domain": item.get("domain", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                })
        
        logger.info(f"✓ Trovati {len(organic_results)} risultati organici")
        
        return {
            "keyword": keyword,
            "total_results": result.get("items_count", 0),
            "ai_overview": ai_overview,
            "ai_overview_sources": ai_overview_sources,
            "organic_results": organic_results[:10],  # Top 10
            "has_ai_overview": ai_overview is not None,
        }
    
    def _parse_ai_overview(self, ai_item: Dict) -> Dict:
        """Parse AI Overview item da DataForSEO"""
        
        # Estrai testo e markdown
        text = ai_item.get("text", "")
        
        # Estrai fonti (references)
        sources = []
        references = ai_item.get("references", [])
        
        for ref in references:
            source = {
                "url": ref.get("url", ""),
                "domain": ref.get("domain", ""),
                "title": ref.get("title", ""),
            }
            sources.append(source)
        
        # Estrai domande correlate (fan-out queries)
        fan_out_queries = []
        items = ai_item.get("items", [])
        for sub_item in items:
            if sub_item.get("type") == "ai_overview_item":
                title = sub_item.get("title", "")
                if title:
                    fan_out_queries.append(title)
        
        return {
            "text": text,
            "sources": sources,
            "fan_out_queries": fan_out_queries,
            "total_sources": len(sources),
        }
    
    def _empty_serp_result(self, keyword: str) -> Dict:
        """Restituisce struttura vuota quando non ci sono risultati"""
        return {
            "keyword": keyword,
            "total_results": 0,
            "ai_overview": None,
            "ai_overview_sources": [],
            "organic_results": [],
            "has_ai_overview": False,
        }
    
    def check_credentials(self) -> bool:
        """Verifica che le credenziali siano valide"""
        try:
            # Usa endpoint ping per verificare auth
            response = requests.get(
                f"{self.base_url}/v3/appendix/user_data",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status_code") == 20000:
                    logger.info("✓ Credenziali DataForSEO valide")
                    return True
            
            logger.error("✗ Credenziali DataForSEO non valide")
            return False
            
        except Exception as e:
            logger.error(f"✗ Errore verifica credenziali: {str(e)}")
            return False

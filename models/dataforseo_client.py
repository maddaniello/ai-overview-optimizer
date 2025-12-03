"""
DataForSEO API Client
"""
import requests
import base64
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from utils.rate_limiter import RateLimiter
from config import DATAFORSEO_API_URL, DATAFORSEO_RATE_LIMIT, LOCATION_CODES, LANGUAGE_CODES


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
    def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        """
        Effettua richiesta POST a DataForSEO API

        Args:
            endpoint: Endpoint API (es. "/serp/google/organic/live/advanced")
            payload: Payload JSON della richiesta

        Returns:
            Response JSON
        """
        url = f"{self.api_url}{endpoint}"

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
                    logger.info("Richiesta completata con successo")
                    return task
                else:
                    error_msg = task.get("status_message", "Unknown error")
                    logger.error(f"Errore DataForSEO: {error_msg}")
                    raise Exception(f"DataForSEO Error: {error_msg}")

            raise Exception("No tasks in response")

        except requests.exceptions.RequestException as e:
            logger.error(f"Errore connessione: {str(e)}")
            raise

    def get_serp_with_ai_overview(
        self,
        keyword: str,
        location: str = "Italy",
        language: str = "Italian",
        device: str = "desktop",
        location_code: int = None,
        language_code: str = None
    ) -> Dict[str, Any]:
        """
        Recupera SERP completo con AI Overview espanso

        Args:
            keyword: Keyword da cercare
            location: Località (es. "Italy")
            language: Lingua (es. "Italian")
            device: Dispositivo ("desktop" o "mobile")
            location_code: Codice località (override di location)
            language_code: Codice lingua (override di language)

        Returns:
            Dict con dati SERP e AI Overview
        """
        # Usa codes se passati, altrimenti lookup da nomi
        loc_code = location_code if location_code else LOCATION_CODES.get(location, 2380)
        lang_code = language_code if language_code else LANGUAGE_CODES.get(language, "it")

        payload = {
            "keyword": keyword,
            "location_code": loc_code,
            "language_code": lang_code,
            "device": device,
            "os": "windows" if device == "desktop" else "android",
            "depth": 100,  # Recupera fino a 100 risultati
            "expand_ai_overview": True,  # CRITICAL: espande contenuto AI Overview
        }

        logger.info(f"Ricerca SERP per: '{keyword}' | Location: {location} ({loc_code}) | Language: {language} ({lang_code})")

        result = self._make_request("/serp/google/organic/live/advanced", payload)

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

        # Debug: log tutti i tipi di item trovati
        item_types = [item.get("type") for item in items]
        logger.info(f"Tipi di item trovati: {set(item_types)}")

        # Estrai AI Overview - cerca vari possibili nomi
        ai_overview = None
        ai_overview_sources = []

        for item in items:
            item_type = item.get("type", "").lower()

            # Cerca AI Overview con vari nomi possibili
            if item_type in ["ai_overview", "ai-overview", "google_ai_overview", "featured_snippet_ai"]:
                logger.info(f"Trovato AI Overview con type: {item_type}")
                ai_overview = self._parse_ai_overview(item)
                ai_overview_sources = ai_overview.get("sources", [])
                logger.info(f"AI Overview trovato con {len(ai_overview_sources)} fonti")
                break

            # Alcuni risultati potrebbero essere nested in featured_snippet
            if item_type == "featured_snippet":
                # Controlla se contiene AI Overview data
                if item.get("text") and len(item.get("text", "")) > 200:
                    logger.info("Trovato featured_snippet con testo lungo - potrebbe essere AI Overview")
                    ai_overview = self._parse_ai_overview(item)
                    ai_overview_sources = ai_overview.get("sources", [])
                    break

        if not ai_overview:
            logger.warning(f"AI Overview NON trovato. Tipi disponibili: {set(item_types)}")
            # Debug: stampa primi 2 item per capire la struttura
            for i, item in enumerate(items[:3]):
                logger.debug(f"Item {i}: type={item.get('type')}, keys={list(item.keys())[:10]}")

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

        logger.info(f"Trovati {len(organic_results)} risultati organici")

        return {
            "keyword": keyword,
            "total_results": result.get("items_count", 0),
            "ai_overview": ai_overview,
            "ai_overview_sources": ai_overview_sources,
            "organic_results": organic_results[:10],  # Top 10
            "has_ai_overview": ai_overview is not None,
            "_debug_item_types": list(set(item_types)),  # Per debug
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
                f"{self.api_url}/appendix/user_data",
                headers=self.headers,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status_code") == 20000:
                    logger.info("Credenziali DataForSEO valide")
                    return True

            logger.error("Credenziali DataForSEO non valide")
            return False

        except Exception as e:
            logger.error(f"Errore verifica credenziali: {str(e)}")
            return False

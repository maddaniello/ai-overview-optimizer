"""
Client per Reranking - supporta Jina AI e Google Vertex AI
"""
import requests
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import get_logger
from config import jina_config, google_config, app_config

logger = get_logger(__name__)

class RerankerClient:
    """Client unificato per reranking con Jina o Google Vertex AI"""
    
    def __init__(self, provider: str = None):
        """
        Args:
            provider: "jina" o "google" (se None usa config)
        """
        self.provider = provider or app_config.reranker_provider
        
        if self.provider == "jina":
            self._init_jina()
        elif self.provider == "google":
            self._init_google()
        else:
            raise ValueError(f"Provider non supportato: {self.provider}")
        
        logger.info(f"Reranker inizializzato: {self.provider}")
    
    def _init_jina(self):
        """Inizializza Jina Reranker"""
        if not jina_config.api_key:
            raise ValueError("JINA_API_KEY non configurata")
        
        self.api_key = jina_config.api_key
        self.base_url = jina_config.base_url
        self.model = jina_config.reranker_model
    
    def _init_google(self):
        """Inizializza Google Vertex AI"""
        if not google_config.project_id:
            raise ValueError("GOOGLE_PROJECT_ID non configurato")
        
        # Import opzionale di Google Cloud
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            self.client = discoveryengine.RankServiceClient()
            self.project_id = google_config.project_id
            self.model = google_config.ranking_model
        except ImportError:
            raise ImportError("google-cloud-discoveryengine non installato")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        return_documents: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rerank documenti per rilevanza rispetto alla query
        
        Args:
            query: Query di riferimento
            documents: Lista di testi da rankare
            return_documents: Se True include testo nei risultati
            
        Returns:
            Lista di dict con score e indice (e opzionalmente testo)
        """
        logger.info(f"📊 Reranking {len(documents)} documenti con {self.provider}")
        
        if self.provider == "jina":
            return self._rerank_jina(query, documents, return_documents)
        else:
            return self._rerank_google(query, documents, return_documents)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _rerank_jina(
        self,
        query: str,
        documents: List[str],
        return_documents: bool
    ) -> List[Dict[str, Any]]:
        """Rerank con Jina AI"""
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": return_documents
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Formatta risultati
            formatted_results = []
            for result in results:
                item = {
                    "index": result.get("index", 0),
                    "relevance_score": result.get("relevance_score", 0.0)
                }
                
                if return_documents:
                    item["document"] = result.get("document", {}).get("text", "")
                
                formatted_results.append(item)
            
            logger.info(f"✓ Reranking Jina completato: {len(formatted_results)} risultati")
            return formatted_results
            
        except Exception as e:
            logger.error(f"✗ Errore reranking Jina: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _rerank_google(
        self,
        query: str,
        documents: List[str],
        return_documents: bool
    ) -> List[Dict[str, Any]]:
        """Rerank con Google Vertex AI"""
        
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            
            # Prepara records
            records = [
                discoveryengine.RankingRecord(
                    id=str(i),
                    content=doc
                )
                for i, doc in enumerate(documents)
            ]
            
            # Crea request
            ranking_config = f"projects/{self.project_id}/locations/global/rankingConfigs/default_ranking_config"
            
            request = discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model=self.model,
                query=query,
                records=records,
                top_n=len(records)
            )
            
            # Esegui ranking
            response = self.client.rank(request=request)
            
            # Formatta risultati
            formatted_results = []
            for record in response.records:
                item = {
                    "index": int(record.id),
                    "relevance_score": record.score
                }
                
                if return_documents:
                    item["document"] = documents[int(record.id)]
                
                formatted_results.append(item)
            
            # Ordina per score decrescente
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info(f"✓ Reranking Google completato: {len(formatted_results)} risultati")
            return formatted_results
            
        except Exception as e:
            logger.error(f"✗ Errore reranking Google: {str(e)}")
            raise
    
    def rerank_with_metadata(
        self,
        query: str,
        items: List[Dict[str, Any]],
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Rerank items con metadata preservando tutti i campi
        
        Args:
            query: Query di riferimento
            items: Lista di dict con metadata
            text_field: Nome campo che contiene il testo da rankare
            
        Returns:
            Lista items ordinati per rilevanza con score aggiunto
        """
        # Estrai testi
        texts = [item.get(text_field, "") for item in items]
        
        # Rerank
        results = self.rerank(query, texts, return_documents=False)
        
        # Aggiungi score agli items
        for result in results:
            idx = result["index"]
            items[idx]["relevance_score"] = result["relevance_score"]
        
        # Ordina per score
        items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return items

"""
Client per Reranking - supporta Jina AI
"""
import requests
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from config import JINA_MODEL, JINA_API_URL


class RerankerClient:
    """Client per reranking con Jina AI"""

    def __init__(self, provider: str = "jina", jina_api_key: str = None):
        """
        Args:
            provider: "jina" (default)
            jina_api_key: Jina API key
        """
        self.provider = provider

        if self.provider == "jina":
            self._init_jina(jina_api_key)
        else:
            raise ValueError(f"Provider non supportato: {self.provider}")

        logger.info(f"Reranker inizializzato: {self.provider}")

    def _init_jina(self, api_key: str = None):
        """Inizializza Jina Reranker"""
        if not api_key:
            raise ValueError("JINA_API_KEY non configurata")

        self.api_key = api_key
        self.base_url = JINA_API_URL
        self.model = JINA_MODEL

    def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]] = None,
        documents: List[str] = None,
        return_documents: bool = False,
        text_field: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Rerank documenti per rilevanza rispetto alla query

        Args:
            query: Query di riferimento
            items: Lista di dict con metadata (usa text_field per il testo)
            documents: Lista di testi da rankare (alternativa a items)
            return_documents: Se True include testo nei risultati
            text_field: Nome campo che contiene il testo (per items)

        Returns:
            Lista di dict con score e indice (e opzionalmente testo)
        """
        # Se passati items con metadata, estrai testi
        if items is not None:
            documents = [item.get(text_field, "") for item in items]
            return self._rerank_with_metadata(query, documents, items, return_documents)

        if documents is None:
            raise ValueError("Devi passare 'items' o 'documents'")

        logger.info(f"Reranking {len(documents)} documenti con {self.provider}")

        return self._rerank_jina(query, documents, return_documents)

    def _rerank_with_metadata(
        self,
        query: str,
        documents: List[str],
        items: List[Dict[str, Any]],
        return_documents: bool
    ) -> List[Dict[str, Any]]:
        """Rerank preservando metadata originali"""
        results = self._rerank_jina(query, documents, return_documents)

        # Arricchisci risultati con metadata originali
        enriched = []
        for result in results:
            idx = result["index"]
            item = items[idx].copy()
            item["relevance_score"] = result["relevance_score"]
            enriched.append(item)

        return enriched

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

            logger.info(f"Reranking Jina completato: {len(formatted_results)} risultati")
            return formatted_results

        except Exception as e:
            logger.error(f"Errore reranking Jina: {str(e)}")
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
        results = self._rerank_jina(query, texts, return_documents=False)

        # Aggiungi score agli items
        for result in results:
            idx = result["index"]
            items[idx]["relevance_score"] = result["relevance_score"]

        # Ordina per score
        items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return items

"""
Client per Reranking - supporta Jina AI o OpenAI Embeddings
"""
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from config import JINA_MODEL, JINA_API_URL, OPENAI_EMBEDDING_MODEL


class RerankerClient:
    """Client per reranking con Jina AI o OpenAI Embeddings"""

    def __init__(
        self,
        provider: str = "embeddings",
        jina_api_key: str = None,
        openai_api_key: str = None
    ):
        """
        Args:
            provider: "jina" o "embeddings" (default: embeddings)
            jina_api_key: Jina API key (solo per provider jina)
            openai_api_key: OpenAI API key (per provider embeddings)
        """
        self.provider = provider
        self.openai_client = None
        self.embeddings_model = OPENAI_EMBEDDING_MODEL

        if self.provider == "jina":
            self._init_jina(jina_api_key)
        elif self.provider == "embeddings":
            self._init_embeddings(openai_api_key)
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

    def _init_embeddings(self, api_key: str = None):
        """Inizializza reranker basato su OpenAI Embeddings"""
        if not api_key:
            raise ValueError("OPENAI_API_KEY non configurata per embeddings reranker")

        from openai import OpenAI
        self.openai_client = OpenAI(api_key=api_key)
        logger.info(f"Embeddings reranker: {self.embeddings_model}")

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

        if self.provider == "embeddings":
            return self._rerank_embeddings(query, documents, return_documents)
        else:
            return self._rerank_jina(query, documents, return_documents)

    def _rerank_with_metadata(
        self,
        query: str,
        documents: List[str],
        items: List[Dict[str, Any]],
        return_documents: bool
    ) -> List[Dict[str, Any]]:
        """Rerank preservando metadata originali"""
        if self.provider == "embeddings":
            results = self._rerank_embeddings(query, documents, return_documents)
        else:
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
    def _rerank_embeddings(
        self,
        query: str,
        documents: List[str],
        return_documents: bool
    ) -> List[Dict[str, Any]]:
        """Rerank usando OpenAI Embeddings + cosine similarity"""
        try:
            # Genera embeddings per query e documenti
            all_texts = [query] + documents
            response = self.openai_client.embeddings.create(
                input=all_texts,
                model=self.embeddings_model
            )

            embeddings = [np.array(data.embedding) for data in response.data]
            query_embedding = embeddings[0]
            doc_embeddings = embeddings[1:]

            # Calcola cosine similarity per ogni documento
            results = []
            for i, doc_emb in enumerate(doc_embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_emb)

                item = {
                    "index": i,
                    "relevance_score": float(similarity)
                }

                if return_documents:
                    item["document"] = documents[i]

                results.append(item)

            # Ordina per score decrescente
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

            logger.info(f"Reranking Embeddings completato: {len(results)} risultati")
            return results

        except Exception as e:
            logger.error(f"Errore reranking Embeddings: {str(e)}")
            raise

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calcola cosine similarity tra due vettori"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

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
        if self.provider == "embeddings":
            results = self._rerank_embeddings(query, texts, return_documents=False)
        else:
            results = self._rerank_jina(query, texts, return_documents=False)

        # Aggiungi score agli items
        for result in results:
            idx = result["index"]
            items[idx]["relevance_score"] = result["relevance_score"]

        # Ordina per score
        items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return items

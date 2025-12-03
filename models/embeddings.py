"""
Client per OpenAI Embeddings e calcolo similarità semantica
"""
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import logger
from utils.rate_limiter import RateLimiter
from config import OPENAI_EMBEDDING_MODEL


class EmbeddingsClient:
    """Client per generazione embeddings e calcolo similarità"""

    def __init__(self, api_key: str):
        """
        Inizializza client con API key

        Args:
            api_key: OpenAI API key
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY non configurata")

        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_EMBEDDING_MODEL

        logger.info(f"Embeddings Client inizializzato: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding per un testo

        Args:
            text: Testo da embeddare

        Returns:
            Array numpy con embedding
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )

            embedding = response.data[0].embedding
            return np.array(embedding)

        except Exception as e:
            logger.error(f"Errore generazione embedding: {str(e)}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Genera embeddings per multipli testi

        Args:
            texts: Lista di testi

        Returns:
            Lista di array numpy
        """
        logger.info(f"Generazione embeddings per {len(texts)} testi")

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [np.array(data.embedding) for data in response.data]

            logger.info(f"Embeddings generati: {len(embeddings)}")
            return embeddings

        except Exception as e:
            logger.error(f"Errore generazione batch embeddings: {str(e)}")
            # Fallback: genera uno alla volta
            logger.warning("Tentativo generazione singola")
            return [self.get_embedding(text) for text in texts]

    # Alias per compatibilità
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Alias per get_embeddings_batch"""
        return self.get_embeddings_batch(texts)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calcola cosine similarity tra due vettori

        Args:
            vec_a: Primo vettore
            vec_b: Secondo vettore

        Returns:
            Score di similarità (0-1)
        """
        # Normalizza vettori
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Calcola cosine similarity
        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)

        return float(similarity)

    @staticmethod
    def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calcola distanza euclidea tra due vettori

        Args:
            vec_a: Primo vettore
            vec_b: Secondo vettore

        Returns:
            Distanza euclidea
        """
        return float(np.linalg.norm(vec_a - vec_b))

    def calculate_similarities(
        self,
        target_text: str,
        comparison_texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Calcola similarità tra un testo target e multipli testi di confronto

        Args:
            target_text: Testo di riferimento
            comparison_texts: Testi da confrontare

        Returns:
            Lista di dict con testo e score similarità
        """
        logger.info(f"Calcolo similarità: 1 target vs {len(comparison_texts)} testi")

        # Genera embeddings
        all_texts = [target_text] + comparison_texts
        embeddings = self.get_embeddings_batch(all_texts)

        target_embedding = embeddings[0]
        comparison_embeddings = embeddings[1:]

        # Calcola similarità
        results = []
        for i, comp_emb in enumerate(comparison_embeddings):
            similarity = self.cosine_similarity(target_embedding, comp_emb)

            results.append({
                "text": comparison_texts[i],
                "similarity": similarity,
                "index": i
            })

        # Ordina per similarità decrescente
        results.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(f"Similarità calcolate - Range: {results[-1]['similarity']:.3f} - {results[0]['similarity']:.3f}")

        return results

    def calculate_similarity_matrix(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Calcola matrice di similarità tra tutti i testi

        Args:
            texts: Lista di testi

        Returns:
            Matrice numpy NxN con similarità
        """
        n = len(texts)
        logger.info(f"Calcolo matrice similarità {n}x{n}")

        # Genera embeddings
        embeddings = self.get_embeddings_batch(texts)

        # Calcola matrice
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    matrix[i][j] = sim
                    matrix[j][i] = sim

        logger.info("Matrice similarità calcolata")

        return matrix

    def find_most_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Trova i K testi più simili alla query

        Args:
            query_text: Testo query
            candidate_texts: Testi candidati
            top_k: Numero di risultati da restituire

        Returns:
            Lista dei top K testi con score
        """
        results = self.calculate_similarities(query_text, candidate_texts)
        return results[:top_k]

"""
Google Ranking API Client
Uses Discovery Engine API for semantic reranking
"""
from typing import List, Dict, Optional
from utils.logger import logger


class GoogleRankingClient:
    """
    Client for Google's Vertex AI Ranking API.
    Provides semantic reranking of documents based on query relevance.

    Pricing: $1 per 1,000 queries (up to 100 documents per query)
    Free tier: 10,000 queries/month
    """

    def __init__(self, project_id: str, credentials_json: Optional[str] = None):
        """
        Initialize the Google Ranking client.

        Args:
            project_id: Google Cloud project ID
            credentials_json: Optional path to service account JSON or JSON string
        """
        self.project_id = project_id
        self.client = None
        self.available = False

        try:
            self._init_client(credentials_json)
        except Exception as e:
            logger.warning(f"Google Ranking API non disponibile: {e}")

    def _init_client(self, credentials_json: Optional[str] = None):
        """Initialize the Discovery Engine client"""
        try:
            from google.cloud import discoveryengine_v1 as discoveryengine
            from google.oauth2 import service_account
            import json

            # Setup credentials if provided
            if credentials_json:
                if credentials_json.startswith('{'):
                    # JSON string
                    creds_dict = json.loads(credentials_json)
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self.client = discoveryengine.RankServiceClient(credentials=credentials)
                else:
                    # File path
                    credentials = service_account.Credentials.from_service_account_file(credentials_json)
                    self.client = discoveryengine.RankServiceClient(credentials=credentials)
            else:
                # Use default credentials (Application Default Credentials)
                self.client = discoveryengine.RankServiceClient()

            self.available = True
            logger.info("Google Ranking API inizializzata")

        except ImportError:
            logger.warning("google-cloud-discoveryengine non installato")
            raise
        except Exception as e:
            logger.error(f"Errore inizializzazione Google Ranking: {e}")
            raise

    def rank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_n: Optional[int] = None,
        model: str = "semantic-ranker-default@latest"
    ) -> List[Dict]:
        """
        Rank documents by relevance to query.

        Args:
            query: The search query
            documents: List of dicts with 'id', 'title', and/or 'content' keys
            top_n: Optional limit on returned results
            model: Ranking model to use

        Returns:
            List of ranked documents with scores (0-1)
        """
        if not self.available:
            logger.warning("Google Ranking non disponibile, usando fallback")
            return self._fallback_rank(documents)

        try:
            from google.cloud import discoveryengine_v1 as discoveryengine

            # Build ranking config path
            ranking_config = self.client.ranking_config_path(
                project=self.project_id,
                location="global",
                ranking_config="default_ranking_config"
            )

            # Prepare records
            records = []
            for i, doc in enumerate(documents[:200]):  # Max 200 per request
                record = discoveryengine.RankingRecord(
                    id=doc.get('id', str(i)),
                    title=doc.get('title', ''),
                    content=doc.get('content', '')[:10000]  # Limit content length
                )
                records.append(record)

            # Create request
            request = discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model=model,
                query=query,
                records=records,
                top_n=top_n
            )

            # Execute ranking
            response = self.client.rank(request=request)

            # Parse results
            results = []
            for record in response.records:
                results.append({
                    'id': record.id,
                    'score': record.score,  # 0-1 relevance score
                    'title': record.title,
                    'content': record.content
                })

            logger.info(f"Google Ranking completato: {len(results)} risultati")
            return results

        except Exception as e:
            logger.error(f"Errore Google Ranking: {e}")
            return self._fallback_rank(documents)

    def _fallback_rank(self, documents: List[Dict]) -> List[Dict]:
        """Fallback ranking (returns documents as-is with placeholder scores)"""
        results = []
        for i, doc in enumerate(documents):
            results.append({
                'id': doc.get('id', str(i)),
                'score': 0.5,  # Neutral score
                'title': doc.get('title', ''),
                'content': doc.get('content', '')
            })
        return results

    def rank_for_ai_overview(
        self,
        ai_overview_text: str,
        contents: List[Dict],
        keyword: str
    ) -> List[Dict]:
        """
        Specialized ranking for AI Overview optimization.
        Ranks contents by similarity to AI Overview text.

        Args:
            ai_overview_text: Google's AI Overview text (reference)
            contents: List of content dicts with 'label', 'content', 'type' keys
            keyword: The search keyword

        Returns:
            Ranked list with scores
        """
        # Use AI Overview as the query (what we want to match)
        query = f"{keyword}: {ai_overview_text[:500]}"

        # Prepare documents
        documents = []
        for i, content in enumerate(contents):
            documents.append({
                'id': content.get('label', f'doc_{i}'),
                'title': content.get('label', ''),
                'content': content.get('content', '')[:5000]
            })

        # Rank
        ranked = self.rank(query, documents)

        # Merge scores back with original content info
        results = []
        for ranked_doc in ranked:
            # Find original content
            original = next(
                (c for c in contents if c.get('label') == ranked_doc['id']),
                None
            )
            if original:
                results.append({
                    **original,
                    'score': ranked_doc['score'],
                    'rank': len(results) + 1
                })

        # Sort by score descending
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return results


# Singleton instance
_google_ranking_client = None


def get_google_ranking_client(
    project_id: str = None,
    credentials_json: str = None
) -> Optional[GoogleRankingClient]:
    """Get or create Google Ranking client singleton"""
    global _google_ranking_client

    if _google_ranking_client is None and project_id:
        try:
            _google_ranking_client = GoogleRankingClient(project_id, credentials_json)
        except Exception:
            pass

    return _google_ranking_client

"""
Optimization Controller - Orchestratore workflow completo
"""
import asyncio
from typing import Dict, Any, Optional, List
from models.dataforseo_client import DataForSEOClient
from models.scraper import ContentScraper
from models.reranker_client import RerankerClient
from models.embeddings import EmbeddingsClient
from models.content_analyzer import ContentAnalyzer
from utils.logger import logger
from config import LOCATION_CODES, LANGUAGE_CODES

class OptimizationController:
    """Controller principale per workflow ottimizzazione"""
    
    def __init__(
        self,
        dataforseo_login: str,
        dataforseo_password: str,
        openai_api_key: str,
        jina_api_key: str = None,
        reranker_provider: str = "embeddings"
    ):
        """
        Inizializza controller con credenziali utente

        Args:
            dataforseo_login: DataForSEO login
            dataforseo_password: DataForSEO password
            openai_api_key: OpenAI API key
            jina_api_key: Jina API key (opzionale)
            reranker_provider: "embeddings" (default) o "jina"
        """
        logger.info("Inizializzazione OptimizationController")

        # Initialize clients with user credentials
        self.dataforseo = DataForSEOClient(
            login=dataforseo_login,
            password=dataforseo_password
        )

        self.scraper = ContentScraper()

        # Usa embeddings come default (no API extra), fallback a jina se disponibile
        self.reranker = RerankerClient(
            provider=reranker_provider,
            jina_api_key=jina_api_key if reranker_provider == "jina" else None,
            openai_api_key=openai_api_key if reranker_provider == "embeddings" else None
        )

        self.embeddings = EmbeddingsClient(api_key=openai_api_key)

        self.analyzer = ContentAnalyzer(openai_api_key=openai_api_key)

        logger.success("Controller inizializzato con successo")
    
    def optimize_content(
        self,
        target_url: str,
        keyword: str,
        location: str = "Italy",
        language: str = "Italian",
        max_sources: int = 5
    ) -> Dict[str, Any]:
        """
        Esegue workflow completo di ottimizzazione
        
        Args:
            target_url: URL pagina da ottimizzare
            keyword: Keyword target
            location: Location name (es: "Italy")
            language: Language name (es: "Italian")
            max_sources: Numero massimo fonti da analizzare
            
        Returns:
            Dict con risultati analisi completa
        """
        try:
            # Converti location e language in codes
            location_code = LOCATION_CODES.get(location, 2380)
            language_code = LANGUAGE_CODES.get(language, "it")
            
            logger.info(f"Inizio ottimizzazione per: {keyword}")
            logger.info(f"Target URL: {target_url}")
            logger.info(f"Location: {location} ({location_code})")
            logger.info(f"Language: {language} ({language_code})")
            
            # === STEP 1: Recupera SERP e AI Overview ===
            logger.info("STEP 1/7: Recupero AI Overview...")
            serp_data = self.dataforseo.get_serp_with_ai_overview(
                keyword=keyword,
                location_code=location_code,
                language_code=language_code
            )
            
            if not serp_data or "ai_overview" not in serp_data or serp_data["ai_overview"] is None:
                logger.warning("AI Overview non trovato per questa keyword")
                return {
                    "success": False,
                    "error": "AI Overview non disponibile per questa keyword"
                }

            ai_overview = serp_data["ai_overview"]
            sources_list = ai_overview.get("sources", []) or []
            logger.success(f"AI Overview trovato con {len(sources_list)} fonti")
            
            # === STEP 2: Scrape target URL ===
            logger.info("STEP 2/7: Scraping target URL...")
            target_answer = self.scraper.extract_answer(target_url, keyword)
            
            if not target_answer:
                logger.error("Impossibile estrarre contenuto da target URL")
                return {
                    "success": False,
                    "error": "Impossibile estrarre contenuto dalla pagina target"
                }
            
            logger.success(f"Target content estratto: {len(target_answer.split())} parole")
            
            # === STEP 3: Scrape fonti AI Overview ===
            logger.info("STEP 3/7: Scraping fonti competitor...")
            
            sources_urls = [
                source.get("url", "") for source in sources_list[:max_sources]
                if source and source.get("url")
            ]
            
            if not sources_urls:
                logger.warning("Nessuna fonte trovata in AI Overview")
                sources_answers = []
            else:
                # Usa asyncio per scraping parallelo
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                sources_answers = loop.run_until_complete(
                    self.scraper.scrape_multiple_async(sources_urls, keyword)
                )
                loop.close()
            
            logger.success(f"{len(sources_answers)} fonti scraped con successo")
            
            # === STEP 4: Calcola relevance scores ===
            logger.info("STEP 4/7: Calcolo relevance scores...")
            
            # Prepara items per reranking
            all_answers = [{"text": target_answer, "url": target_url, "is_target": True}]
            
            for source in sources_answers:
                all_answers.append({
                    "text": source["answer"],
                    "url": source["url"],
                    "is_target": False
                })
            
            # Rerank
            ranked_answers = self.reranker.rerank(
                query=keyword,
                items=all_answers
            )
            
            # Separa target da sources
            target_result = next((r for r in ranked_answers if r.get("is_target")), None)
            top_sources = [r for r in ranked_answers if not r.get("is_target")][:5]
            
            if not target_result:
                logger.error("Target non trovato nei risultati reranking")
                return {"success": False, "error": "Errore nel calcolo relevance score"}
            
            target_score = target_result["relevance_score"]
            logger.success(f"Target relevance score: {target_score:.3f}")
            
            # === STEP 5: Calcola similarità semantica ===
            logger.info("STEP 5/7: Calcolo similarità semantica...")
            
            # Genera embeddings
            all_texts = [target_answer] + [s["text"] for s in top_sources]
            embeddings = self.embeddings.generate_embeddings_batch(all_texts)
            
            # Calcola similarità target vs sources
            target_emb = embeddings[0]
            sources_emb = embeddings[1:]
            
            similarities = []
            for src_emb in sources_emb:
                sim = self.embeddings.cosine_similarity(target_emb, src_emb)
                similarities.append(sim)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            logger.success(f"Similarità semantica media: {avg_similarity:.3f}")
            
            # === STEP 6: Analisi entità ===
            logger.info("STEP 6/7: Analisi gap entità...")
            
            gap_analysis = self.analyzer.analyze_entity_gap(
                target_text=target_answer,
                competitor_texts=[s["text"] for s in top_sources[:3]]
            )
            
            missing_entities = gap_analysis["missing_entities"]
            entity_coverage = gap_analysis["entity_coverage"]
            
            logger.success(f"Entity gap: {len(missing_entities)} entità mancanti")
            logger.info(f"Entity coverage: {entity_coverage:.2%}")
            
            # === STEP 7: Ottimizzazione LLM ===
            logger.info("STEP 7/7: Generazione versione ottimizzata...")
            
            optimized_result = self.analyzer.optimize_answer(
                query=keyword,
                current_answer=target_answer,
                top_sources=top_sources[:3],
                missing_entities=missing_entities
            )

            if not optimized_result:
                optimized_answer = target_answer
                optimized_score = target_score
                improvement = 0
                logger.warning("Ottimizzazione non disponibile, uso risposta originale")
            else:
                optimized_answer = optimized_result.get("optimized_answer", target_answer)

                # Calcola nuovo score per versione ottimizzata
                optimized_items = [
                    {"text": optimized_answer, "is_optimized": True},
                    *[{"text": s["text"], "is_optimized": False} for s in top_sources]
                ]

                optimized_ranked = self.reranker.rerank(keyword, optimized_items)
                optimized_score = next(
                    (r["relevance_score"] for r in optimized_ranked if r.get("is_optimized")),
                    target_score
                )

                improvement = ((optimized_score - target_score) / target_score * 100) if target_score > 0 else 0

                logger.success(f"Ottimizzazione completata: +{improvement:.1f}%")
            
            # === STEP 8: Fan-out queries ===
            fan_out_queries = ai_overview.get("fan_out_queries", []) or []
            
            # === RISULTATI FINALI ===
            result = {
                "success": True,
                "analysis": {
                    "target_url": target_url,
                    "keyword": keyword,
                    "location": location,
                    "language": language,
                    "current_relevance_score": float(target_score),
                    "is_ai_overview_source": target_url in [s.get("url", "") for s in sources_list],
                    "ai_overview_sources_count": len(sources_list)
                },
                "target_content": {
                    "answer": target_answer,
                    "word_count": len(target_answer.split()),
                    "char_count": len(target_answer)
                },
                "top_sources": [
                    {
                        "url": s["url"],
                        "domain": s["url"].split("/")[2] if "/" in s["url"] else s["url"],
                        "relevance_score": float(s["relevance_score"]),
                        "semantic_similarity": float(similarities[i]) if i < len(similarities) else 0,
                        "answer_preview": s["text"][:200] + "..."
                    }
                    for i, s in enumerate(top_sources)
                ],
                "gap_analysis": {
                    "missing_entities": missing_entities,
                    "entity_coverage": float(entity_coverage),
                    "semantic_similarity_avg": float(avg_similarity),
                    "relevance_gap": float(max(s["relevance_score"] for s in top_sources) - target_score) if top_sources else 0
                },
                "optimized_answer": {
                    "text": optimized_answer,
                    "new_relevance_score": float(optimized_score),
                    "improvement_percentage": float(improvement),
                    "word_count": len(optimized_answer.split()),
                    "char_count": len(optimized_answer)
                },
                "fan_out_opportunities": [
                    {
                        "query": query,
                        "opportunity_score": "HIGH" if query.lower() not in target_answer.lower() else "LOW"
                    }
                    for query in fan_out_queries[:5]
                ],
                "recommendations": self._generate_recommendations(
                    target_score=target_score,
                    optimized_score=optimized_score,
                    missing_entities=missing_entities,
                    fan_out_queries=fan_out_queries
                )
            }
            
            logger.success("Analisi completata con successo!")
            return result
            
        except Exception as e:
            logger.error(f"Errore durante ottimizzazione: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_recommendations(
        self,
        target_score: float,
        optimized_score: float,
        missing_entities: List[tuple],
        fan_out_queries: List[str]
    ) -> List[str]:
        """Genera raccomandazioni actionable"""
        recommendations = []
        
        # Score-based
        if target_score < 0.5:
            recommendations.append("⚠️ Score molto basso: considera riscrittura completa del contenuto")
        elif target_score < 0.7:
            recommendations.append("📝 Integra le entità mancanti e aggiungi più dettagli")
        
        # Entities
        if len(missing_entities) > 5:
            top_entities = [ent["entity"] for ent in missing_entities[:5]]
            recommendations.append(f"🏷️ Aggiungi queste entità chiave: {', '.join(top_entities)}")
        
        # Improvement
        if optimized_score > target_score:
            recommendations.append(f"✅ Usa la versione ottimizzata (+{((optimized_score - target_score) / target_score * 100):.1f}%)")
        
        # Fan-out
        if len(fan_out_queries) > 0:
            recommendations.append(f"❓ Crea sezioni FAQ per le {len(fan_out_queries)} query correlate")
        
        return recommendations

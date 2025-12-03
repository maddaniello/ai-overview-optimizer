"""
Optimization Controller - Orchestratore workflow completo
"""
from typing import Dict, Any, Optional
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
        jina_api_key: str,
        reranker_provider: str = "jina"
    ):
        """
        Inizializza controller con credenziali utente
        
        Args:
            dataforseo_login: DataForSEO login
            dataforseo_password: DataForSEO password
            openai_api_key: OpenAI API key
            jina_api_key: Jina API key
            reranker_provider: "jina" o "google"
        """
        logger.info("Inizializzazione OptimizationController")
        
        # Initialize clients with user credentials
        self.dataforseo = DataForSEOClient(
            login=dataforseo_login,
            password=dataforseo_password
        )
        
        self.scraper = ContentScraper()
        
        self.reranker = RerankerClient(
            provider=reranker_provider,
            jina_api_key=jina_api_key if reranker_provider == "jina" else None
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
        # ... resto del codice rimane uguale
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 INIZIO OTTIMIZZAZIONE")
        logger.info(f"URL: {target_url}")
        logger.info(f"Keyword: {keyword}")
        logger.info(f"{'='*80}\n")
        
        results = {
            "metadata": {
                "target_url": target_url,
                "keyword": keyword,
                "location": location,
                "language": language,
                "timestamp": datetime.now().isoformat(),
            },
            "serp_data": None,
            "target_analysis": None,
            "sources_analysis": None,
            "gap_analysis": None,
            "optimization": None,
            "fan_out_analysis": None,
            "recommendations": []
        }
        
        try:
            # STEP 1: Recupera SERP e AI Overview
            logger.info("📋 STEP 1: Recupero SERP e AI Overview")
            serp_data = self.dataforseo.get_serp_with_ai_overview(
                keyword=keyword,
                location=location,
                language=language
            )
            results["serp_data"] = serp_data
            
            if not serp_data["has_ai_overview"]:
                logger.warning("⚠️ Nessun AI Overview trovato per questa keyword")
                results["recommendations"].append(
                    "❌ Questa keyword non ha AI Overview attivo - potrebbe non essere una query informativa"
                )
                # Continua comunque con analisi organica
            
            # STEP 2: Scrape target URL
            logger.info("\n📋 STEP 2: Scraping URL target")
            target_content = await self.scraper.scrape_url(target_url, keyword)
            
            # Estrai risposta dal contenuto target
            target_answer = target_content.get("answer_text") or \
                           self.scraper.extract_answer_from_content(
                               target_content.get("content", ""),
                               keyword,
                               max_words=300
                           )
            
            results["target_analysis"] = {
                "url": target_url,
                "domain": extract_domain(target_url),
                "answer": target_answer,
                "content_length": len(target_content.get("content", "")),
                "quality_metrics": self.analyzer.analyze_content_quality(target_answer)
            }
            
            logger.info(f"✓ Target scrapato: {len(target_answer)} caratteri risposta")
            
            # STEP 3: Scrape AI Overview sources
            logger.info("\n📋 STEP 3: Scraping fonti AI Overview")
            
            sources_to_scrape = []
            if serp_data["has_ai_overview"]:
                # Usa fonti AI Overview
                sources_to_scrape = [
                    src["url"] for src in serp_data["ai_overview_sources"][:max_sources]
                ]
            else:
                # Fallback: usa top risultati organici
                sources_to_scrape = [
                    res["url"] for res in serp_data["organic_results"][:max_sources]
                ]
            
            logger.info(f"Scraping {len(sources_to_scrape)} fonti")
            
            sources_contents = await self.scraper.scrape_multiple_urls(
                sources_to_scrape,
                extract_answer_for_query=keyword
            )
            
            # Estrai risposte dalle fonti
            sources_data = []
            for i, content in enumerate(sources_contents):
                answer = content.get("answer_text") or \
                        self.scraper.extract_answer_from_content(
                            content.get("content", ""),
                            keyword,
                            max_words=300
                        )
                
                if answer and len(answer) > 50:  # Filtra risposte troppo corte
                    sources_data.append({
                        "url": content["url"],
                        "domain": extract_domain(content["url"]),
                        "answer": answer,
                        "position": i + 1
                    })
            
            logger.info(f"✓ Estratte {len(sources_data)} risposte valide")
            
            # STEP 4: Calcola rilevanza con Reranker
            logger.info("\n📋 STEP 4: Calcolo rilevanza contestuale")
            
            all_answers = [target_answer] + [src["answer"] for src in sources_data]
            
            reranked = self.reranker.rerank(
                query=keyword,
                documents=all_answers,
                return_documents=False
            )
            
            # Assegna score
            target_relevance_score = reranked[0]["relevance_score"]
            results["target_analysis"]["relevance_score"] = target_relevance_score
            
            for i, result in enumerate(reranked[1:]):
                sources_data[i]["relevance_score"] = result["relevance_score"]
            
            # Ordina sources per score
            sources_data.sort(key=lambda x: x["relevance_score"], reverse=True)
            results["sources_analysis"] = sources_data
            
            logger.info(f"✓ Target relevance score: {target_relevance_score:.3f}")
            logger.info(f"✓ Best competitor score: {sources_data[0]['relevance_score']:.3f}")
            
            # STEP 5: Calcola similarità semantica
            logger.info("\n📋 STEP 5: Analisi similarità semantica")
            
            similarities = self.embeddings.calculate_similarities(
                target_answer,
                [src["answer"] for src in sources_data]
            )
            
            # Aggiungi similarità ai sources
            for i, sim in enumerate(similarities):
                sources_data[i]["semantic_similarity"] = sim["similarity"]
            
            avg_similarity = sum(s["similarity"] for s in similarities) / len(similarities)
            logger.info(f"✓ Similarità semantica media: {avg_similarity:.3f}")
            
            # STEP 6: Analisi gap entità
            logger.info("\n📋 STEP 6: Analisi gap entità")
            
            gap_analysis = self.analyzer.analyze_entity_gap(
                target_answer,
                [src["answer"] for src in sources_data[:3]],  # Top 3
                top_n=10
            )
            
            results["gap_analysis"] = gap_analysis
            logger.info(f"✓ Trovate {len(gap_analysis['missing_entities'])} entità mancanti")
            
            # STEP 7: Genera ottimizzazione
            logger.info("\n📋 STEP 7: Generazione risposta ottimizzata")
            
            optimized_answer = self.analyzer.generate_optimized_answer(
                query=keyword,
                current_answer=target_answer,
                top_sources=sources_data[:3],
                missing_entities=gap_analysis["missing_entities"],
                max_words=300
            )
            
            if optimized_answer:
                # Re-calcola score per versione ottimizzata
                optimized_score_result = self.reranker.rerank(
                    query=keyword,
                    documents=[optimized_answer] + [src["answer"] for src in sources_data],
                    return_documents=False
                )
                
                optimized_score = optimized_score_result[0]["relevance_score"]
                improvement = calculate_improvement_percentage(
                    target_relevance_score,
                    optimized_score
                )
                
                results["optimization"] = {
                    "optimized_answer": optimized_answer,
                    "new_relevance_score": optimized_score,
                    "previous_score": target_relevance_score,
                    "improvement_percentage": improvement,
                    "word_count": len(optimized_answer.split()),
                    "quality_metrics": self.analyzer.analyze_content_quality(optimized_answer)
                }
                
                logger.info(f"✓ Score ottimizzato: {optimized_score:.3f} ({improvement:+.1f}%)")
            else:
                results["optimization"] = None
                logger.warning("⚠️ Impossibile generare ottimizzazione")
            
            # STEP 8: Analisi fan-out queries (opzionale)
            if include_fan_out and serp_data["has_ai_overview"]:
                logger.info("\n📋 STEP 8: Analisi fan-out queries")
                
                fan_out_queries = serp_data["ai_overview"].get("fan_out_queries", [])
                
                if fan_out_queries:
                    fan_out_analysis = await self._analyze_fan_out_queries(
                        target_url,
                        fan_out_queries[:5]  # Max 5 per non esagerare
                    )
                    results["fan_out_analysis"] = fan_out_analysis
                    logger.info(f"✓ Analizzate {len(fan_out_analysis)} fan-out queries")
            
            # STEP 9: Genera raccomandazioni
            logger.info("\n📋 STEP 9: Generazione raccomandazioni")
            recommendations = self._generate_recommendations(results)
            results["recommendations"] = recommendations
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ OTTIMIZZAZIONE COMPLETATA")
            logger.info(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"✗ Errore durante ottimizzazione: {str(e)}")
            raise
    
    async def _analyze_fan_out_queries(
        self,
        target_url: str,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """Analizza se target URL è presente nelle fan-out queries"""
        
        results = []
        
        for query in queries:
            try:
                # Recupera SERP per query correlata (senza AI Overview per velocità)
                serp = self.dataforseo.get_serp_with_ai_overview(
                    keyword=query,
                    location="Italy",
                    language="Italian"
                )
                
                # Check se target URL è nei risultati
                is_ranking = any(
                    target_url in result["url"]
                    for result in serp["organic_results"]
                )
                
                # Trova posizione se presente
                position = None
                if is_ranking:
                    for result in serp["organic_results"]:
                        if target_url in result["url"]:
                            position = result["position"]
                            break
                
                results.append({
                    "query": query,
                    "is_ranking": is_ranking,
                    "position": position,
                    "opportunity": not is_ranking,
                    "total_results": serp["total_results"]
                })
                
            except Exception as e:
                logger.warning(f"Errore analisi fan-out '{query}': {str(e)}")
        
        return results
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Genera raccomandazioni basate sull'analisi"""
        
        recommendations = []
        
        # Raccomandazioni based on relevance score
        target_score = results["target_analysis"].get("relevance_score", 0)
        best_competitor_score = results["sources_analysis"][0]["relevance_score"] if results["sources_analysis"] else 0
        
        if target_score < 0.7:
            recommendations.append(
                f"⚠️ Relevance score basso ({target_score:.2f}) - la risposta non è ottimizzata per la query"
            )
        
        if target_score < best_competitor_score:
            gap = (best_competitor_score - target_score) / best_competitor_score * 100
            recommendations.append(
                f"📊 Gap di rilevanza: {gap:.1f}% rispetto al miglior competitor"
            )
        
        # Raccomandazioni based on entities
        gap_analysis = results.get("gap_analysis", {})
        missing_entities = gap_analysis.get("missing_entities", [])
        
        if missing_entities:
            top_missing = ", ".join([ent["entity"] for ent in missing_entities[:5]])
            recommendations.append(
                f"🏷️ Aggiungi entità chiave: {top_missing}"
            )
        
        # Raccomandazioni based on fan-out
        fan_out = results.get("fan_out_analysis", [])
        opportunities = [f for f in fan_out if f.get("opportunity")]
        
        if opportunities:
            recommendations.append(
                f"💡 {len(opportunities)} opportunità fan-out query non coperte"
            )
            for opp in opportunities[:3]:
                recommendations.append(
                    f"  → Crea sezione per: '{opp['query']}'"
                )
        
        # Raccomandazioni based on optimization
        optimization = results.get("optimization")
        if optimization and optimization.get("improvement_percentage", 0) > 10:
            recommendations.append(
                f"✅ Usa la versione ottimizzata (+{optimization['improvement_percentage']:.1f}% rilevanza)"
            )
        
        return recommendations

"""
Orchestrator Agent - Coordina il workflow completo multi-agente
"""
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from .base_agent import BaseAgent, AgentState, AgentStatus
from .llm_client import LLMClient
from utils.logger import logger

from models.dataforseo_client import DataForSEOClient
from models.scraper import ContentScraper
from models.embeddings import EmbeddingsClient


class OrchestratorAgent(BaseAgent):
    """
    Agente orchestratore che coordina l'intero workflow.
    Gestisce gli altri agenti e mantiene lo stato condiviso.
    """

    def __init__(self, log_callback: Callable = None):
        super().__init__(
            name="orchestrator",
            description="Coordina il workflow completo di ottimizzazione",
            log_callback=log_callback
        )
        self.state = None
        self.llm_client = None
        self.dataforseo = None
        self.scraper = None
        self.embeddings = None

    def initialize_clients(self, state: AgentState):
        """Inizializza tutti i client necessari"""
        self.log("Inizializzazione client...", level="info")

        # LLM Client
        self.llm_client = LLMClient(
            model_id=state.model_id,
            openai_api_key=state.openai_api_key,
            gemini_api_key=state.gemini_api_key
        )

        # DataForSEO
        self.dataforseo = DataForSEOClient(
            login=state.dataforseo_login,
            password=state.dataforseo_password
        )

        # Scraper
        self.scraper = ContentScraper()

        # Embeddings (sempre OpenAI)
        if state.openai_api_key:
            self.embeddings = EmbeddingsClient(api_key=state.openai_api_key)

        self.log("Client inizializzati", level="success")

    async def execute(self, state: AgentState) -> AgentState:
        """Esegue il workflow completo"""
        self.state = state
        self.initialize_clients(state)

        # === FASE 1: RECUPERO DATI SERP ===
        state = await self._phase_serp_retrieval(state)
        if not state.serp_data:
            return state

        # === FASE 2: SCRAPING CONTENUTI ===
        state = await self._phase_content_scraping(state)

        # === FASE 3: RANKING INIZIALE ===
        state = await self._phase_initial_ranking(state)

        # === FASE 4: CICLO OTTIMIZZAZIONE ===
        state = await self._phase_optimization_loop(state)

        # === FASE 5: ANALISI STRATEGICA ===
        state = await self._phase_strategic_analysis(state)

        # === FASE 6: PIANO CONTENUTO ===
        state = await self._phase_content_planning(state)

        self.log("Workflow completato!", level="success")
        return state

    async def _phase_serp_retrieval(self, state: AgentState) -> AgentState:
        """Fase 1: Recupero dati SERP e AI Overview"""
        self.log("=" * 50, level="info")
        self.log("FASE 1: Recupero dati SERP", level="info")
        self.log("=" * 50, level="info")

        try:
            from config import LOCATION_CODES, LANGUAGE_CODES

            location_code = LOCATION_CODES.get(state.location, 2380)
            language_code = LANGUAGE_CODES.get(state.language, "it")

            self.log(f"Ricerca per: '{state.keyword}'", level="info")
            self.log(f"Location: {state.location} ({location_code})", level="info")

            # Recupera SERP
            serp_data = self.dataforseo.get_serp_with_ai_overview(
                keyword=state.keyword,
                location_code=location_code,
                language_code=language_code
            )

            if not serp_data:
                self.log("Nessun dato SERP trovato", level="error")
                return state

            state.serp_data = serp_data

            # Estrai AI Overview
            ai_overview = serp_data.get("ai_overview")
            if ai_overview:
                state.ai_overview_text = ai_overview.get("text", "")
                state.ai_overview_sources = ai_overview.get("sources", [])
                state.fan_out_queries = ai_overview.get("fan_out_queries", [])

                self.log(f"AI Overview trovato: {len(state.ai_overview_text)} caratteri", level="success")
                self.log(f"Fonti AIO: {len(state.ai_overview_sources)}", level="info")
                self.log(f"Fan-out queries: {len(state.fan_out_queries)}", level="info")

                # Log preview AI Overview
                if state.ai_overview_text:
                    preview = state.ai_overview_text[:200] + "..." if len(state.ai_overview_text) > 200 else state.ai_overview_text
                    self.log(f"Preview AIO: {preview}", level="info", data={"text": state.ai_overview_text})
            else:
                self.log("AI Overview non disponibile per questa keyword", level="warning")

            # Risultati organici
            state.organic_results = serp_data.get("organic_results", [])[:state.max_serp_results]
            self.log(f"Risultati organici: {len(state.organic_results)}", level="info")

        except Exception as e:
            self.log(f"Errore recupero SERP: {e}", level="error")

        return state

    async def _phase_content_scraping(self, state: AgentState) -> AgentState:
        """Fase 2: Scraping contenuti target e competitor"""
        self.log("=" * 50, level="info")
        self.log("FASE 2: Scraping Contenuti", level="info")
        self.log("=" * 50, level="info")

        # Scrape target URL se fornito
        if state.target_url:
            self.log(f"Scraping URL target: {state.target_url}", level="info")
            try:
                target_content = self.scraper.extract_answer(state.target_url, state.keyword)
                if target_content:
                    state.target_content = target_content
                    self.log(f"Target scraped: {len(target_content.split())} parole", level="success")
                else:
                    self.log("Impossibile estrarre contenuto target", level="warning")
            except Exception as e:
                self.log(f"Errore scraping target: {e}", level="error")

        # Se l'utente ha fornito la sua risposta, usa quella
        if state.user_answer:
            state.current_answer = state.user_answer
            self.log("Usando risposta fornita dall'utente", level="info")
        elif state.target_content:
            state.current_answer = state.target_content
            self.log("Usando contenuto estratto dal target URL", level="info")

        # Scrape fonti AI Overview
        if state.ai_overview_sources:
            self.log(f"Scraping {len(state.ai_overview_sources[:state.max_sources])} fonti AIO...", level="info")

            urls = [s.get("url") for s in state.ai_overview_sources[:state.max_sources] if s.get("url")]

            for i, url in enumerate(urls):
                self.log(f"  [{i+1}/{len(urls)}] {url[:60]}...", level="info")
                try:
                    content = self.scraper.extract_answer(url, state.keyword)
                    if content:
                        source_info = state.ai_overview_sources[i] if i < len(state.ai_overview_sources) else {}

                        # Estrai una "risposta" rilevante dal contenuto (primi 500 caratteri significativi)
                        response_preview = content[:500].strip()
                        if len(content) > 500:
                            # Trova un punto di interruzione naturale
                            last_period = response_preview.rfind('.')
                            if last_period > 200:
                                response_preview = response_preview[:last_period + 1]

                        state.competitor_contents.append({
                            "url": url,
                            "domain": source_info.get("domain", url.split("/")[2] if "/" in url else url),
                            "title": source_info.get("title", ""),
                            "content": content,
                            "response_preview": response_preview,  # Preview della risposta
                            "source_type": "ai_overview",
                            "word_count": len(content.split())
                        })
                        self.log(f"    ✓ {len(content.split())} parole", level="success")
                    else:
                        self.log(f"    ✗ Contenuto vuoto", level="warning")
                except Exception as e:
                    self.log(f"    ✗ Errore: {e}", level="error")

            self.log(f"Fonti scraped: {len(state.competitor_contents)}/{len(urls)}", level="info")

        return state

    async def _phase_initial_ranking(self, state: AgentState) -> AgentState:
        """Fase 3: Calcolo ranking iniziale"""
        self.log("=" * 50, level="info")
        self.log("FASE 3: Ranking Iniziale", level="info")
        self.log("=" * 50, level="info")

        if not self.embeddings:
            self.log("Embeddings non disponibili, skip ranking", level="warning")
            return state

        try:
            # Prepara testi per embedding
            texts_to_embed = []
            text_metadata = []

            # AI Overview come riferimento
            if state.ai_overview_text:
                texts_to_embed.append(state.ai_overview_text)
                text_metadata.append({
                    "type": "reference",
                    "label": "AI Overview Google",
                    "url": None,
                    "is_reference": True
                })
                self.log("Riferimento: AI Overview di Google", level="info")

            # Risposta corrente utente (se presente)
            if state.current_answer:
                texts_to_embed.append(state.current_answer)
                text_metadata.append({
                    "type": "user_answer",
                    "label": "La Tua Risposta",
                    "url": state.target_url,
                    "is_reference": False
                })
                self.log(f"Tua risposta: {len(state.current_answer.split())} parole", level="info")

            # Competitor da AI Overview
            for i, comp in enumerate(state.competitor_contents):
                # Usa response_preview se disponibile, altrimenti content troncato
                content_for_embedding = comp.get("response_preview", comp["content"][:500])
                texts_to_embed.append(content_for_embedding)
                text_metadata.append({
                    "type": "competitor",
                    "label": f"Competitor: {comp['domain']}",
                    "url": comp["url"],
                    "domain": comp["domain"],
                    "title": comp.get("title", ""),
                    "response_preview": comp.get("response_preview", content_for_embedding[:200]),
                    "is_reference": False
                })
                self.log(f"Competitor {i+1}: {comp['domain']} ({comp['word_count']} parole)", level="info")

            if len(texts_to_embed) < 2:
                self.log("Non abbastanza testi per ranking", level="warning")
                return state

            # Genera embeddings
            self.log(f"Generazione embeddings per {len(texts_to_embed)} contenuti...", level="info")
            embeddings = self.embeddings.get_embeddings_batch(texts_to_embed)
            self.log("Embeddings generati", level="success")

            # Calcola similarità vs AI Overview (primo elemento)
            import numpy as np

            reference_idx = 0  # AI Overview è sempre il riferimento
            reference_emb = np.array(embeddings[reference_idx])

            ranking = []
            for i, (emb, meta) in enumerate(zip(embeddings, text_metadata)):
                if meta.get("is_reference"):
                    # AI Overview è il riferimento, score = 1.0
                    score = 1.0
                else:
                    emb_arr = np.array(emb)
                    score = float(np.dot(reference_emb, emb_arr) / (np.linalg.norm(reference_emb) * np.linalg.norm(emb_arr)))

                ranking_item = {
                    "rank": 0,  # Sarà calcolato dopo
                    "type": meta["type"],
                    "label": meta["label"],
                    "url": meta.get("url"),
                    "score": round(score, 4),
                    "preview": texts_to_embed[i][:200] + "..." if len(texts_to_embed[i]) > 200 else texts_to_embed[i],
                    "is_reference": meta.get("is_reference", False)
                }

                # Aggiungi response_preview per competitor
                if meta["type"] == "competitor":
                    ranking_item["response_preview"] = meta.get("response_preview", "")
                    ranking_item["domain"] = meta.get("domain", "")
                    ranking_item["title"] = meta.get("title", "")

                ranking.append(ranking_item)

            # Ordina per score (escluso reference che è sempre primo)
            ranking.sort(key=lambda x: (not x.get("is_reference", False), -x["score"]))
            for i, item in enumerate(ranking):
                item["rank"] = i + 1

            state.initial_ranking = ranking
            state.current_ranking = [r.copy() for r in ranking]

            # Log ranking dettagliato
            self.log("=" * 40, level="info")
            self.log("RANKING INIZIALE (vs AI Overview):", level="info")
            self.log("=" * 40, level="info")

            for item in ranking:
                type_emoji = {
                    "reference": "🎯",
                    "user_answer": "📝",
                    "competitor": "🏢"
                }.get(item["type"], "•")

                self.log(
                    f"  #{item['rank']} {type_emoji} {item['label']}: {item['score']:.4f}",
                    level="success" if item["type"] == "user_answer" else "info"
                )

            # Trova e salva score utente iniziale
            user_item = next((r for r in ranking if r["type"] == "user_answer"), None)
            if user_item:
                state.best_score = user_item["score"]
                user_rank = user_item["rank"]
                total_competitors = len([r for r in ranking if r["type"] == "competitor"])
                self.log(f"\n📊 La tua posizione: #{user_rank} su {total_competitors + 1} contenuti", level="info")
                self.log(f"📈 Score iniziale: {state.best_score:.4f}", level="info")

        except Exception as e:
            self.log(f"Errore calcolo ranking: {e}", level="error")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", level="error")

        return state

    async def _phase_optimization_loop(self, state: AgentState) -> AgentState:
        """Fase 4: Ciclo di ottimizzazione iterativo"""
        self.log("=" * 50, level="info")
        self.log(f"FASE 4: Ciclo Ottimizzazione ({state.max_iterations} iterazioni)", level="info")
        self.log("=" * 50, level="info")

        if not state.current_answer:
            self.log("Nessuna risposta da ottimizzare", level="warning")
            return state

        current_answer = state.current_answer
        best_answer = current_answer
        best_score = state.best_score

        for iteration in range(1, state.max_iterations + 1):
            self.log(f"\n--- ITERAZIONE {iteration}/{state.max_iterations} ---", level="info")

            # Genera ottimizzazione con ragionamento
            optimization_result = await self._optimize_single_iteration(
                state=state,
                current_answer=current_answer,
                iteration=iteration
            )

            if not optimization_result:
                self.log(f"Iterazione {iteration} fallita", level="warning")
                continue

            optimized_answer = optimization_result["answer"]
            reasoning = optimization_result["reasoning"]

            self.log(f"Ragionamento:\n{reasoning[:300]}...", level="info")

            # Calcola nuovo score
            new_score = await self._calculate_score(state, optimized_answer)

            # Aggiorna ranking
            new_ranking = await self._update_ranking(state, optimized_answer, iteration)

            # Log risultato iterazione
            improvement = ((new_score - best_score) / best_score * 100) if best_score > 0 else 0
            self.log(f"Score iterazione {iteration}: {new_score:.4f} (Δ {improvement:+.2f}%)", level="info")

            # Salva iterazione
            iteration_data = {
                "iteration": iteration,
                "answer": optimized_answer,
                "reasoning": reasoning,
                "score": new_score,
                "improvement": improvement,
                "ranking": new_ranking
            }
            state.iterations.append(iteration_data)

            # Aggiorna se migliore
            if new_score > best_score:
                best_answer = optimized_answer
                best_score = new_score
                self.log(f"Nuovo miglior score: {best_score:.4f}", level="success")

            # Usa risposta ottimizzata per prossima iterazione
            current_answer = optimized_answer
            state.current_ranking = new_ranking

        state.best_answer = best_answer
        state.best_score = best_score
        self.log(f"\nMiglior risposta finale - Score: {best_score:.4f}", level="success")

        return state

    async def _optimize_single_iteration(
        self,
        state: AgentState,
        current_answer: str,
        iteration: int
    ) -> Optional[Dict]:
        """Esegue una singola iterazione di ottimizzazione"""
        try:
            # Costruisci contesto
            context_parts = []

            if state.ai_overview_text:
                context_parts.append(f"AI OVERVIEW GOOGLE (riferimento da battere):\n{state.ai_overview_text}")

            if state.competitor_contents:
                comp_text = "\n\n".join([
                    f"COMPETITOR {i+1} ({c['domain']}):\n{c.get('response_preview', c['content'][:300])}"
                    for i, c in enumerate(state.competitor_contents[:3])
                ])
                context_parts.append(f"RISPOSTE DEI COMPETITOR:\n{comp_text}")

            # Aggiungi storia delle iterazioni precedenti se non è la prima
            if iteration > 1 and state.iterations:
                prev_iterations = []
                for prev in state.iterations[-2:]:  # Ultime 2 iterazioni
                    prev_iterations.append(
                        f"Iterazione {prev['iteration']} (score: {prev['score']:.4f}):\n"
                        f"Ragionamento: {prev['reasoning'][:200]}..."
                    )
                if prev_iterations:
                    context_parts.append(f"ITERAZIONI PRECEDENTI:\n" + "\n\n".join(prev_iterations))

            context = "\n\n---\n\n".join(context_parts)

            # Prompt diverso per prima iterazione vs successive
            if iteration == 1:
                # Se c'è una risposta iniziale, ottimizzala; altrimenti creane una nuova
                if current_answer and len(current_answer.strip()) > 50:
                    prompt = f"""KEYWORD: {state.keyword}

CONTENUTO ATTUALE DA CUI PARTIRE (per riferimento, NON copiare):
{current_answer[:1000]}{'...' if len(current_answer) > 1000 else ''}

COMPITO: Scrivi una NUOVA risposta ottimizzata per Google AI Overview.

REQUISITI:
1. Crea contenuto ORIGINALE ispirandoti all'AI Overview di Google
2. NON copiare il contenuto attuale - usalo solo come riferimento
3. Copri i punti chiave che Google considera importanti
4. Massimo 250-300 parole
5. Stile: chiaro, autorevole, diretto, utile
6. Formato: paragrafi fluidi senza elenchi puntati o titoli

OUTPUT: Scrivi SOLO la risposta ottimizzata, niente altro."""
                else:
                    prompt = f"""KEYWORD: {state.keyword}

COMPITO: Scrivi una risposta ottimizzata per comparire in Google AI Overview.

REQUISITI:
1. Rispondi in modo diretto e completo alla query "{state.keyword}"
2. Copri tutti i punti chiave presenti nell'AI Overview di Google
3. Sii più completo e autorevole dei competitor
4. Massimo 250-300 parole
5. Stile: chiaro, diretto, professionale
6. Formato: paragrafi fluidi senza elenchi puntati o titoli

OUTPUT: Scrivi SOLO la risposta ottimizzata, niente altro."""
            else:
                prev_score = state.iterations[-1]['score'] if state.iterations else 0
                prompt = f"""KEYWORD: {state.keyword}

ITERAZIONE {iteration}/{state.max_iterations}

RISPOSTA PRECEDENTE (score: {prev_score:.4f}):
{current_answer}

COMPITO: Migliora questa risposta per aumentare lo score di similarità con l'AI Overview.

STRATEGIA DI MIGLIORAMENTO:
1. Analizza cosa manca rispetto all'AI Overview di Google
2. Aggiungi informazioni chiave non presenti
3. Migliora la struttura e la chiarezza
4. Mantieni massimo 250-300 parole

IMPORTANTE:
- Migliora la risposta esistente, non riscriverla completamente
- Concentrati sui gap semantici con l'AI Overview
- Ogni iterazione deve portare un miglioramento concreto

OUTPUT: Scrivi SOLO la risposta migliorata, niente altro."""

            self.log(f"Generando ottimizzazione iterazione {iteration}...", level="info")

            result = self.llm_client.generate_with_reasoning(
                prompt=prompt,
                context=context,
                task_description=f"Ottimizzazione iterazione {iteration}",
                temperature=0.7 if iteration == 1 else 0.5  # Meno creatività nelle iterazioni successive
            )

            return result

        except Exception as e:
            self.log(f"Errore ottimizzazione: {e}", level="error")
            return None

    async def _calculate_score(self, state: AgentState, answer: str) -> float:
        """Calcola score di similarità con AI Overview"""
        if not self.embeddings or not state.ai_overview_text:
            return 0.5

        try:
            import numpy as np

            embeddings = self.embeddings.get_embeddings_batch([state.ai_overview_text, answer])
            ref_emb = np.array(embeddings[0])
            ans_emb = np.array(embeddings[1])

            score = float(np.dot(ref_emb, ans_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(ans_emb)))
            return score

        except Exception as e:
            self.log(f"Errore calcolo score: {e}", level="error")
            return 0.5

    async def _update_ranking(self, state: AgentState, new_answer: str, iteration: int) -> List[Dict]:
        """Aggiorna ranking con nuova risposta"""
        if not state.current_ranking:
            return []

        try:
            # Calcola score per nuova risposta
            new_score = await self._calculate_score(state, new_answer)

            # Crea nuovo ranking
            new_ranking = [r.copy() for r in state.current_ranking if r["type"] != "optimized"]

            new_ranking.append({
                "rank": 0,
                "type": "optimized",
                "label": f"Ottimizzata (iter. {iteration})",
                "url": state.target_url,
                "score": round(new_score, 4),
                "preview": new_answer[:150] + "..."
            })

            # Riordina
            new_ranking.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(new_ranking):
                item["rank"] = i + 1

            return new_ranking

        except Exception as e:
            self.log(f"Errore aggiornamento ranking: {e}", level="error")
            return state.current_ranking

    async def _phase_strategic_analysis(self, state: AgentState) -> AgentState:
        """Fase 5: Analisi strategica"""
        self.log("=" * 50, level="info")
        self.log("FASE 5: Analisi Strategica", level="info")
        self.log("=" * 50, level="info")

        try:
            context = f"""
KEYWORD: {state.keyword}
AI OVERVIEW: {state.ai_overview_text[:500] if state.ai_overview_text else 'Non disponibile'}
FONTI AIO: {len(state.ai_overview_sources)}
RISULTATI ORGANICI: {len(state.organic_results)}
SCORE MIGLIORE: {state.best_score:.4f}
"""

            prompt = """Genera un'analisi strategica completa che includa:

1. INTENTO DI RICERCA
- Intento primario (informativo, transazionale, navigazionale)
- Domande implicite risolte dall'AI Overview
- Target audience

2. COMPETITOR ANALYSIS
- Punti di forza delle fonti AI Overview
- Debolezze e gap identificati
- Opportunità non sfruttate

3. CONCETTI CHIAVE
- Definizioni fondamentali da includere
- Entità e termini rilevanti
- Argomenti correlati

Rispondi in formato JSON strutturato."""

            result = self.llm_client.generate_with_reasoning(
                prompt=prompt,
                context=context,
                task_description="Analisi strategica",
                temperature=0.5
            )

            state.strategic_analysis = {
                "reasoning": result["reasoning"],
                "analysis": result["answer"],
                "generated_at": datetime.now().isoformat()
            }

            self.log("Analisi strategica completata", level="success")

        except Exception as e:
            self.log(f"Errore analisi strategica: {e}", level="error")

        return state

    async def _phase_content_planning(self, state: AgentState) -> AgentState:
        """Fase 6: Piano contenuto"""
        self.log("=" * 50, level="info")
        self.log("FASE 6: Piano Contenuto", level="info")
        self.log("=" * 50, level="info")

        try:
            context = f"""
KEYWORD: {state.keyword}
AI OVERVIEW: {state.ai_overview_text[:500] if state.ai_overview_text else 'Non disponibile'}
RISPOSTA OTTIMIZZATA: {state.best_answer[:500] if state.best_answer else 'Non disponibile'}
FAN-OUT QUERIES: {', '.join(state.fan_out_queries[:5])}
"""

            prompt = """Genera un piano contenuto dettagliato con:

1. STRUTTURA ARTICOLO CONSIGLIATA
Per ogni sezione indica:
- Tag (H1, H2, H3)
- Titolo consigliato
- Contenuto suggerito (includendo dove inserire la risposta ottimizzata)
- Elementi multimediali suggeriti

2. SEZIONI FAQ
- Domande derivate dalle fan-out queries
- Risposte suggerite

3. SUGGERIMENTI AGGIUNTIVI
- Link interni consigliati
- Call-to-action
- Schema markup raccomandato

Formato output: JSON strutturato"""

            result = self.llm_client.generate_with_reasoning(
                prompt=prompt,
                context=context,
                task_description="Piano contenuto",
                temperature=0.6
            )

            state.content_plan = {
                "reasoning": result["reasoning"],
                "plan": result["answer"],
                "generated_at": datetime.now().isoformat()
            }

            self.log("Piano contenuto generato", level="success")

        except Exception as e:
            self.log(f"Errore piano contenuto: {e}", level="error")

        return state

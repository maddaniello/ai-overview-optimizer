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
        self.google_ranking = None

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
            self.log("OpenAI Embeddings client inizializzato", level="info")

        # Google Ranking (opzionale)
        self.log(f"Ranking method richiesto: {state.ranking_method}", level="info")
        self.log(f"Google Project ID: {state.google_project_id or 'Non configurato'}", level="info")

        if state.google_project_id and state.ranking_method == "google":
            try:
                from models.google_ranking import GoogleRankingClient
                self.google_ranking = GoogleRankingClient(
                    project_id=state.google_project_id,
                    credentials_json=state.google_credentials_json
                )
                self.log("✅ Google Discovery Engine Ranking API inizializzata", level="success")
                self.log("Il ranking userà Google Cloud per reranking semantico", level="info")
            except Exception as e:
                self.log(f"⚠️ Google Ranking non disponibile: {e}", level="warning")
                self.log("Fallback a OpenAI Embeddings per ranking", level="info")
        else:
            if state.ranking_method == "embeddings":
                self.log("Ranking method: OpenAI Embeddings (cosine similarity)", level="info")
            else:
                self.log(f"Google Ranking non attivo (project_id={bool(state.google_project_id)}, method={state.ranking_method})", level="info")

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

            # Debug: mostra tipi di item trovati
            if serp_data.get("_debug_item_types"):
                self.log(f"Tipi item DataForSEO: {serp_data['_debug_item_types']}", level="info")

            # Estrai AI Overview
            ai_overview = serp_data.get("ai_overview")

            # Debug: log se AI Overview è stato trovato
            has_aio = serp_data.get("has_ai_overview", False)
            self.log(f"has_ai_overview flag: {has_aio}", level="info")

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
                self.log("⚠️ AI Overview NON trovato nella risposta DataForSEO", level="warning")
                self.log("Verrà generato un riferimento sintetico dai competitor", level="info")

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

        if not self.embeddings and not self.google_ranking:
            self.log("Nessun sistema di ranking disponibile", level="warning")
            return state

        # Se non c'è AI Overview, genera un riferimento sintetico
        if not state.ai_overview_text and state.competitor_contents:
            self.log("AI Overview non disponibile - genero riferimento da competitor", level="warning")
            state = await self._generate_synthetic_reference(state)

        try:
            # Prepara contenuti per ranking
            contents_to_rank = []

            # Risposta corrente utente (se presente)
            if state.current_answer:
                contents_to_rank.append({
                    "type": "user_answer",
                    "label": "La Tua Risposta",
                    "content": state.current_answer,
                    "url": state.target_url
                })
                self.log(f"Tua risposta: {len(state.current_answer.split())} parole", level="info")

            # Competitor da AI Overview
            for i, comp in enumerate(state.competitor_contents):
                content_for_ranking = comp.get("response_preview", comp["content"][:500])
                contents_to_rank.append({
                    "type": "competitor",
                    "label": f"Competitor: {comp['domain']}",
                    "content": content_for_ranking,
                    "url": comp["url"],
                    "domain": comp["domain"],
                    "response_preview": comp.get("response_preview", "")
                })
                self.log(f"Competitor {i+1}: {comp['domain']} ({comp['word_count']} parole)", level="info")

            if len(contents_to_rank) < 1:
                self.log("Nessun contenuto da rankare", level="warning")
                return state

            # Calcola ranking usando il metodo configurato
            ref_type = "AI Overview" if state.ai_overview_text else "Riferimento sintetico"

            if self.google_ranking and self.google_ranking.available:
                self.log(f"🔵 Usando Google Discovery Engine per ranking vs {ref_type}...", level="info")
                ranking = await self._calculate_ranking_google(state, contents_to_rank)
            else:
                self.log(f"🟢 Usando OpenAI Embeddings per ranking vs {ref_type}...", level="info")
                ranking = await self._calculate_ranking_embeddings(state, contents_to_rank)

            state.initial_ranking = ranking
            state.current_ranking = [r.copy() for r in ranking]

            # Log ranking
            self.log("=" * 40, level="info")
            self.log(f"RANKING (similarità vs {ref_type}):", level="info")
            self.log("=" * 40, level="info")

            for item in ranking:
                emoji = "📝" if item["type"] == "user_answer" else "🏢"
                self.log(f"  #{item['rank']} {emoji} {item['label']}: {item['score']:.4f}", level="info")

            # Score iniziale utente
            user_item = next((r for r in ranking if r["type"] == "user_answer"), None)
            if user_item:
                state.best_score = user_item["score"]
                self.log(f"\n📈 Tuo score iniziale: {state.best_score:.4f}", level="success")
            else:
                state.best_score = 0.5  # Default se nessuna risposta utente
                self.log("Nessuna risposta utente - si parte da zero", level="info")

        except Exception as e:
            self.log(f"Errore calcolo ranking: {e}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")

        return state

    async def _generate_synthetic_reference(self, state: AgentState) -> AgentState:
        """Genera un riferimento sintetico quando AI Overview non è disponibile"""
        self.log("Generazione riferimento sintetico basato sui competitor...", level="info")

        try:
            # Raccogli contenuto dai competitor
            competitor_texts = []
            for comp in state.competitor_contents[:3]:
                preview = comp.get("response_preview", comp["content"][:500])
                competitor_texts.append(preview)

            combined_context = "\n\n---\n\n".join(competitor_texts)

            prompt = f"""Basandoti sui contenuti dei competitor per la keyword "{state.keyword}", genera una risposta ideale che potrebbe apparire in un AI Overview di Google.

CONTENUTI COMPETITOR:
{combined_context[:2000]}

Genera una risposta di 200-250 parole che:
1. Sintetizzi le informazioni chiave dei competitor
2. Risponda direttamente alla query "{state.keyword}"
3. Sia strutturata in paragrafi fluidi
4. Abbia tono informativo e autorevole

Scrivi SOLO il testo della risposta:"""

            result = self.llm_client.generate_with_reasoning(
                prompt=prompt,
                context="",
                task_description="Generazione riferimento sintetico",
                temperature=0.5
            )

            if result and result.get("answer"):
                state.ai_overview_text = result["answer"]
                state.synthetic_reference = True  # Flag per indicare che è sintetico
                self.log(f"Riferimento sintetico generato: {len(state.ai_overview_text)} caratteri", level="success")
            else:
                self.log("Impossibile generare riferimento sintetico", level="error")

        except Exception as e:
            self.log(f"Errore generazione riferimento: {e}", level="error")

        return state

    async def _calculate_ranking_google(
        self,
        state: AgentState,
        contents: List[Dict]
    ) -> List[Dict]:
        """Calcola ranking usando Google Discovery Engine"""
        try:
            self.log("Chiamata Google Discovery Engine Ranking API...", level="info")

            # Usa il metodo specializzato per AI Overview
            ranked = self.google_ranking.rank_for_ai_overview(
                ai_overview_text=state.ai_overview_text,
                contents=contents,
                keyword=state.keyword
            )

            self.log(f"✅ Google Ranking completato: {len(ranked)} risultati", level="success")

            # Log dettagliato dei risultati
            for r in ranked[:3]:
                self.log(f"  Google rank: {r.get('label', 'N/A')} -> score {r.get('score', 0):.4f}", level="info")

            return ranked

        except Exception as e:
            self.log(f"❌ Errore Google Ranking: {e}", level="error")
            self.log("Fallback a OpenAI Embeddings...", level="warning")
            return await self._calculate_ranking_embeddings(state, contents)

    async def _calculate_ranking_embeddings(
        self,
        state: AgentState,
        contents: List[Dict]
    ) -> List[Dict]:
        """Calcola ranking usando embeddings OpenAI"""
        import numpy as np

        if not state.ai_overview_text:
            self.log("AI Overview mancante, uso ranking neutro", level="warning")
            return [{"rank": i+1, **c, "score": 0.5} for i, c in enumerate(contents)]

        # Prepara testi: AI Overview + tutti i contenuti
        texts = [state.ai_overview_text] + [c["content"] for c in contents]

        # Genera embeddings
        embeddings = self.embeddings.get_embeddings_batch(texts)

        # Embedding di riferimento (AI Overview)
        ref_emb = np.array(embeddings[0])

        # Calcola cosine similarity per ogni contenuto
        ranking = []
        for i, content in enumerate(contents):
            emb = np.array(embeddings[i + 1])  # +1 perché 0 è AI Overview

            # Cosine similarity
            dot = np.dot(ref_emb, emb)
            norm = np.linalg.norm(ref_emb) * np.linalg.norm(emb)
            score = float(dot / norm) if norm > 0 else 0

            # Score può essere max ~0.95 per contenuti diversi dall'originale
            # Non dovrebbe mai essere 1.0 esatto
            ranking.append({
                "rank": 0,
                "type": content["type"],
                "label": content["label"],
                "url": content.get("url"),
                "content": content["content"][:200],
                "score": round(score, 4),
                "domain": content.get("domain", ""),
                "response_preview": content.get("response_preview", "")
            })

        # Ordina per score
        ranking.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(ranking):
            item["rank"] = i + 1

        return ranking

    async def _phase_optimization_loop(self, state: AgentState) -> AgentState:
        """Fase 4: Ciclo di ottimizzazione iterativo"""
        self.log("=" * 50, level="info")
        self.log(f"FASE 4: Ciclo Ottimizzazione ({state.max_iterations} iterazioni)", level="info")
        self.log("=" * 50, level="info")

        # Se non c'è risposta iniziale, generiamo da zero
        current_answer = state.current_answer or ""
        initial_score = state.best_score
        best_answer = current_answer
        best_score = initial_score

        for iteration in range(1, state.max_iterations + 1):
            self.log(f"\n{'='*30}", level="info")
            self.log(f"ITERAZIONE {iteration}/{state.max_iterations}", level="info")
            self.log(f"{'='*30}", level="info")

            # Genera ottimizzazione
            optimization_result = await self._optimize_single_iteration(
                state=state,
                current_answer=current_answer,
                iteration=iteration,
                current_score=best_score
            )

            if not optimization_result or not optimization_result.get("answer"):
                self.log(f"Iterazione {iteration} fallita - skip", level="warning")
                continue

            optimized_answer = optimization_result["answer"]
            reasoning = optimization_result["reasoning"]

            # Log ragionamento (completo, non troncato)
            self.log(f"Ragionamento: {reasoning}", level="info")

            # Calcola nuovo score
            new_score = await self._calculate_score(state, optimized_answer)

            # Calcola improvement rispetto allo score iniziale (non al best)
            improvement_vs_initial = ((new_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
            improvement_vs_prev = ((new_score - best_score) / best_score * 100) if best_score > 0 else 0

            self.log(f"Score: {new_score:.4f} (vs iniziale: {improvement_vs_initial:+.2f}%)", level="info")

            # Aggiorna ranking
            new_ranking = await self._update_ranking(state, optimized_answer, iteration)

            # Salva iterazione
            iteration_data = {
                "iteration": iteration,
                "answer": optimized_answer,
                "reasoning": reasoning,
                "score": new_score,
                "improvement": improvement_vs_initial,
                "ranking": new_ranking
            }
            state.iterations.append(iteration_data)

            # Aggiorna best se migliore
            if new_score > best_score:
                best_answer = optimized_answer
                best_score = new_score
                self.log(f"✓ Nuovo miglior score: {best_score:.4f}", level="success")
            else:
                self.log(f"Score non migliorato (best: {best_score:.4f})", level="warning")

            # Usa sempre la nuova risposta per la prossima iterazione
            # (anche se peggiore, per esplorare diverse direzioni)
            current_answer = optimized_answer
            state.current_ranking = new_ranking

        state.best_answer = best_answer
        state.best_score = best_score

        total_improvement = ((best_score - initial_score) / initial_score * 100) if initial_score > 0 else 0
        self.log(f"\n{'='*40}", level="success")
        self.log(f"OTTIMIZZAZIONE COMPLETATA", level="success")
        self.log(f"Score finale: {best_score:.4f} ({total_improvement:+.2f}% vs iniziale)", level="success")
        self.log(f"{'='*40}", level="success")

        return state

    async def _optimize_single_iteration(
        self,
        state: AgentState,
        current_answer: str,
        iteration: int,
        current_score: float = 0.5
    ) -> Optional[Dict]:
        """Esegue una singola iterazione di ottimizzazione"""
        try:
            # Il contesto è l'AI Overview - questo è il target da imitare
            ai_overview = state.ai_overview_text or ""

            # Estrai concetti chiave dall'AI Overview per guidare l'ottimizzazione
            key_concepts = self._extract_key_phrases(ai_overview)

            # Prompt specifico per ogni iterazione
            if iteration == 1 and (not current_answer or len(current_answer.strip()) < 50):
                # Prima iterazione senza risposta: genera da zero
                prompt = f"""Scrivi una risposta ottimizzata per la keyword "{state.keyword}".

RIFERIMENTO - Questo è il testo che Google mostra nell'AI Overview:
---
{ai_overview}
---

CONCETTI CHIAVE da includere: {', '.join(key_concepts[:10])}

ISTRUZIONI:
1. Scrivi 200-280 parole in italiano fluente
2. Copri gli stessi argomenti del testo di riferimento
3. Usa terminologia simile ma non copiare letteralmente
4. Scrivi in paragrafi fluidi (NO elenchi puntati, NO titoli, NO markdown)
5. Mantieni tono informativo e autorevole

Scrivi SOLO il testo della risposta, senza introduzioni o commenti."""

            elif iteration == 1:
                # Prima iterazione con risposta esistente: ottimizza
                prompt = f"""Riscrivi questa risposta per renderla più simile allo stile dell'AI Overview di Google.

TESTO GOOGLE AI OVERVIEW (riferimento):
---
{ai_overview}
---

LA TUA RISPOSTA ATTUALE:
---
{current_answer[:1000]}
---

ISTRUZIONI:
1. Mantieni le informazioni corrette della tua risposta
2. Aggiungi concetti presenti nell'AI Overview ma mancanti nella tua risposta
3. Usa terminologia e struttura simile al riferimento Google
4. 200-280 parole, paragrafi fluidi, NO markdown

Scrivi la versione migliorata:"""

            else:
                # Iterazioni successive: migliora progressivamente
                prev_score = state.iterations[-1]['score'] if state.iterations else current_score
                score_pct = prev_score * 100

                # Identifica cosa manca confrontando con AI Overview
                prompt = f"""Migliora questa risposta. Score attuale: {score_pct:.1f}%

OBIETTIVO: Avvicinarsi di più al testo dell'AI Overview di Google.

AI OVERVIEW (target):
---
{ai_overview}
---

RISPOSTA DA MIGLIORARE:
---
{current_answer}
---

COSA FARE:
- Identifica 2-3 concetti/frasi dell'AI Overview non presenti nella risposta
- Riformula per includerli mantenendo naturalezza
- Usa parole chiave simili a quelle di Google
- Mantieni 200-280 parole

Scrivi la versione MIGLIORATA (solo il testo, nient'altro):"""

            self.log(f"Generando risposta iterazione {iteration}...", level="info")

            # Usa temperature crescente per esplorare soluzioni diverse
            temp = 0.5 + (iteration - 1) * 0.1  # 0.5, 0.6, 0.7...
            temp = min(temp, 0.8)  # Max 0.8

            result = self.llm_client.generate_with_reasoning(
                prompt=prompt,
                context="",
                task_description=f"Iterazione {iteration} - Ottimizzazione per AI Overview",
                temperature=temp
            )

            return result

        except Exception as e:
            self.log(f"Errore ottimizzazione: {e}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="error")
            return None

    def _extract_key_phrases(self, text: str, max_phrases: int = 15) -> List[str]:
        """Estrae frasi/concetti chiave da un testo"""
        if not text:
            return []

        import re

        # Pulisci e tokenizza
        text = text.lower()
        # Rimuovi punteggiatura eccetto apostrofi
        text = re.sub(r"[^\w\s']", ' ', text)

        # Stopwords italiane comuni
        stopwords = {
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
            'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
            'che', 'e', 'è', 'sono', 'sia', 'come', 'anche', 'dove',
            'quando', 'perché', 'cosa', 'chi', 'quale', 'quanto',
            'non', 'più', 'molto', 'poco', 'tutto', 'ogni', 'altro',
            'questo', 'quello', 'stesso', 'proprio', 'solo', 'ancora',
            'sempre', 'mai', 'già', 'ora', 'poi', 'quindi', 'però',
            'se', 'o', 'ma', 'mentre', 'dopo', 'prima', 'essere', 'avere',
            'fare', 'può', 'possono', 'deve', 'devono', 'viene', 'vengono'
        }

        words = text.split()

        # Estrai bigrammi e trigrammi significativi
        phrases = []

        # Singole parole significative (>4 caratteri, non stopword)
        for word in words:
            if len(word) > 4 and word not in stopwords:
                phrases.append(word)

        # Bigrammi
        for i in range(len(words) - 1):
            if words[i] not in stopwords or words[i+1] not in stopwords:
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 8:
                    phrases.append(bigram)

        # Conta frequenze
        from collections import Counter
        freq = Counter(phrases)

        # Ritorna i più frequenti
        return [phrase for phrase, _ in freq.most_common(max_phrases)]

    async def _calculate_score(self, state: AgentState, answer: str) -> float:
        """Calcola score di similarità con AI Overview"""
        if not state.ai_overview_text:
            self.log("⚠️ AI Overview mancante per calcolo score", level="warning")
            return 0.5

        # Per il calcolo veloce durante le iterazioni, usiamo sempre embeddings
        # Google Ranking è usato solo per il ranking iniziale completo
        if not self.embeddings:
            self.log("⚠️ Embeddings non disponibili", level="warning")
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

"""
Content Analyzer - Analisi entità e ottimizzazione LLM
"""
import spacy
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
from openai import OpenAI
from utils.logger import logger
from utils.helpers import truncate_text, count_words
from config import OPENAI_CHAT_MODEL, MAX_ANSWER_LENGTH


class ContentAnalyzer:
    """Analizzatore contenuti con NER e LLM"""

    def __init__(self, openai_api_key: str):
        """
        Inizializza analyzer

        Args:
            openai_api_key: OpenAI API key
        """
        # OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

        # SpaCy model
        try:
            self.nlp = spacy.load("it_core_news_sm")
            logger.success("SpaCy model caricato: it_core_news_sm")
        except OSError:
            logger.warning("Modello it_core_news_sm non trovato, scarico...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "it_core_news_sm"])
            self.nlp = spacy.load("it_core_news_sm")

        logger.info("ContentAnalyzer inizializzato")

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Estrae entità nominate dal testo

        Args:
            text: Testo da analizzare

        Returns:
            Lista di entità con tipo e testo
        """
        if not self.nlp:
            logger.warning("spaCy non disponibile - uso estrazione base")
            return self._extract_entities_basic(text)

        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

    def _extract_entities_basic(self, text: str) -> List[Dict[str, str]]:
        """Estrazione entità base senza spaCy (parole capitalizzate)"""
        import re

        # Trova parole capitalizzate (entità potenziali)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(pattern, text)

        entities = []
        for match in matches:
            entities.append({
                "text": match,
                "label": "ENTITY",
                "start": 0,
                "end": 0
            })

        return entities

    def analyze_entity_gap(
        self,
        target_text: str,
        competitor_texts: List[str],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analizza gap di entità tra target e testi di confronto

        Args:
            target_text: Testo da ottimizzare
            competitor_texts: Testi competitor
            top_n: Numero di entità top da restituire

        Returns:
            Dict con analisi gap
        """
        logger.info("Analisi gap entità")

        # Estrai entità dal target
        target_entities = self.extract_entities(target_text)
        target_entity_texts = set(ent["text"].lower() for ent in target_entities)

        logger.info(f"Target: {len(target_entity_texts)} entità uniche")

        # Estrai entità dai competitor
        all_competitor_entities = []
        for comp_text in competitor_texts:
            comp_entities = self.extract_entities(comp_text)
            all_competitor_entities.extend(ent["text"] for ent in comp_entities)

        # Conta frequenze
        entity_counter = Counter(ent.lower() for ent in all_competitor_entities)

        # Identifica entità mancanti nel target
        missing_entities = []
        for entity, count in entity_counter.most_common(top_n * 2):  # Prendi più del necessario
            if entity not in target_entity_texts and count >= 2:  # Presente in almeno 2 competitor
                missing_entities.append({
                    "entity": entity,
                    "frequency": count,
                    "present_in_sources": count
                })

        # Limita a top_n
        missing_entities = missing_entities[:top_n]

        logger.info(f"Trovate {len(missing_entities)} entità mancanti significative")

        return {
            "target_entities": list(target_entity_texts),
            "target_entities_count": len(target_entity_texts),
            "competitor_entities_count": len(set(all_competitor_entities)),
            "missing_entities": missing_entities,
            "entity_coverage": len(target_entity_texts) / max(len(set(all_competitor_entities)), 1)
        }

    def generate_optimized_answer(
        self,
        query: str,
        current_answer: str,
        top_sources: List[Dict[str, Any]],
        missing_entities: List[Dict[str, str]],
        max_words: int = 300
    ) -> Optional[Dict[str, Any]]:
        """
        Genera versione ottimizzata della risposta usando LLM

        Args:
            query: Query originale
            current_answer: Risposta corrente
            top_sources: Top risposte competitor
            missing_entities: Entità mancanti
            max_words: Massimo parole

        Returns:
            Dict con risposta ottimizzata o None se LLM non disponibile
        """
        if not self.openai_client:
            logger.warning("OpenAI non disponibile - skip ottimizzazione LLM")
            return None

        logger.info("Generazione risposta ottimizzata con LLM")

        # Formatta top sources - gestisce sia dict con 'answer' che con 'text'
        sources_text = "\n\n".join([
            f"**Fonte {i+1}** (Score: {src.get('relevance_score', 0):.2f}):\n{truncate_text(src.get('answer', src.get('text', '')), 200)}"
            for i, src in enumerate(top_sources[:3])
        ])

        # Formatta entità mancanti
        entities_text = ", ".join([
            f"{ent['entity']} (freq: {ent['frequency']})"
            for ent in missing_entities[:5]
        ])

        # Crea prompt
        prompt = f"""Query utente: "{query}"

**RISPOSTA CORRENTE** (da ottimizzare):
{current_answer}

**TOP 3 RISPOSTE CONCORRENTI** (più rilevanti secondo reranker):
{sources_text}

**ENTITÀ MANCANTI** (presenti nei competitor):
{entities_text if entities_text else "Nessuna entità mancante significativa"}

**TASK**:
Genera una versione OTTIMIZZATA della risposta corrente che:

1. Mantiene il tono e stile della risposta originale
2. Integra le entità mancanti chiave in modo naturale
3. Copre tutti gli aspetti rilevanti delle risposte competitor
4. Risponde in modo DIRETTO e COMPLETO alla query
5. È concisa: massimo {max_words} parole
6. Include informazioni fattuali e specifiche
7. Usa un linguaggio chiaro e accessibile

**IMPORTANTE**:
- NON copiare frasi dai competitor
- NON aggiungere informazioni non verificabili
- Mantieni lo stesso livello di autorevolezza
- Rispondi nella stessa lingua della query
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Sei un esperto SEO content optimizer. Il tuo compito è migliorare risposte per massimizzare la rilevanza contestuale per AI Overview di Google."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )

            optimized_answer = response.choices[0].message.content.strip()

            # Verifica lunghezza
            word_count = count_words(optimized_answer)
            if word_count > max_words * 1.2:  # Tolleranza 20%
                logger.warning(f"Risposta ottimizzata troppo lunga ({word_count} parole), tronco")
                words = optimized_answer.split()
                optimized_answer = " ".join(words[:max_words]) + "..."

            logger.info(f"Risposta ottimizzata generata: {word_count} parole")

            return {
                "optimized_answer": optimized_answer,
                "word_count": word_count
            }

        except Exception as e:
            logger.error(f"Errore generazione ottimizzazione: {str(e)}")
            return None

    # Alias per compatibilità
    def optimize_answer(
        self,
        query: str,
        current_answer: str,
        top_sources: List[Dict[str, Any]],
        missing_entities: List[Dict[str, str]],
        max_words: int = 300
    ) -> Optional[Dict[str, Any]]:
        """Alias per generate_optimized_answer"""
        return self.generate_optimized_answer(
            query=query,
            current_answer=current_answer,
            top_sources=top_sources,
            missing_entities=missing_entities,
            max_words=max_words
        )

    def analyze_content_quality(self, text: str) -> Dict[str, Any]:
        """
        Analizza qualità del contenuto

        Args:
            text: Testo da analizzare

        Returns:
            Dict con metriche di qualità
        """
        word_count = count_words(text)
        char_count = len(text)

        # Conta frasi (approssimato)
        sentences = text.count('.') + text.count('!') + text.count('?')

        # Parole per frase media
        avg_words_per_sentence = word_count / max(sentences, 1)

        # Readability score approssimato (Flesch-like simplificato)
        avg_word_length = char_count / max(word_count, 1)

        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentences,
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "avg_word_length": round(avg_word_length, 1),
            "readability": "good" if 10 <= avg_words_per_sentence <= 20 else "needs_improvement"
        }

    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """
        Estrae frasi chiave dal testo

        Args:
            text: Testo da analizzare
            top_n: Numero di frasi da estrarre

        Returns:
            Lista di frasi chiave
        """
        if not self.nlp:
            return []

        doc = self.nlp(text)

        # Usa noun chunks come frasi chiave
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Conta frequenze
        phrase_counter = Counter(phrases)

        # Filtra frasi troppo corte o troppo lunghe
        filtered = [
            phrase for phrase, _ in phrase_counter.most_common(top_n * 2)
            if 2 <= len(phrase.split()) <= 5
        ]

        return filtered[:top_n]

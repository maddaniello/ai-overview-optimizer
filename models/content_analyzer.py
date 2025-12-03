"""
Content Analyzer - Analisi entità e ottimizzazione LLM
"""
import re
from collections import Counter
from typing import List, Dict, Optional, Any
from openai import OpenAI
from utils.logger import logger
from utils.helpers import truncate_text, count_words
from config import OPENAI_CHAT_MODEL, MAX_ANSWER_LENGTH

# SpaCy is optional
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy non disponibile - uso estrazione entità base")


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

        # SpaCy model (optional)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("it_core_news_sm")
                logger.success("SpaCy model caricato: it_core_news_sm")
            except OSError:
                logger.warning("Modello spaCy non trovato - uso estrazione base")
                self.nlp = None

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
        """Estrazione entità base senza spaCy (parole capitalizzate + keywords)"""
        entities = []

        # Trova parole capitalizzate (nomi propri)
        pattern = r'\b[A-Z][a-zàèéìòù]+(?:\s+[A-Z][a-zàèéìòù]+)*\b'
        matches = re.findall(pattern, text)

        for match in matches:
            if len(match) > 2:  # Ignora parole troppo corte
                entities.append({
                    "text": match,
                    "label": "ENTITY",
                    "start": 0,
                    "end": 0
                })

        # Estrai anche termini tecnici comuni (parole lunghe)
        words = text.split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 8 and clean_word.isalpha():
                entities.append({
                    "text": clean_word.lower(),
                    "label": "TERM",
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
        for entity, count in entity_counter.most_common(top_n * 2):
            if entity not in target_entity_texts and count >= 2:
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
        max_words: int = 300,
        ai_overview_text: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Genera versione ottimizzata della risposta usando LLM

        Args:
            query: Query/keyword target
            current_answer: Risposta corrente da ottimizzare
            top_sources: Top fonti competitor
            missing_entities: Entità mancanti
            max_words: Limite parole
            ai_overview_text: Testo dell'AI Overview di Google (riferimento principale!)
        """
        if not self.openai_client:
            logger.warning("OpenAI non disponibile - skip ottimizzazione LLM")
            return None

        logger.info("Generazione risposta ottimizzata con LLM")

        # Formatta AI Overview (RIFERIMENTO PRINCIPALE!)
        ai_overview_section = ""
        if ai_overview_text:
            ai_overview_section = f"""
**AI OVERVIEW DI GOOGLE** (questo è il riferimento PRINCIPALE - la tua risposta deve coprire questi punti):
{truncate_text(ai_overview_text, 500)}
"""

        # Formatta top sources
        sources_text = "\n\n".join([
            f"**Fonte {i+1}** (Score: {src.get('relevance_score', 0):.2f}):\n{truncate_text(src.get('answer', src.get('text', '')), 200)}"
            for i, src in enumerate(top_sources[:3])
        ])

        # Formatta entità mancanti
        entities_text = ", ".join([
            f"{ent['entity']} (freq: {ent['frequency']})"
            for ent in missing_entities[:5]
        ])

        prompt = f"""Query utente: "{query}"

**RISPOSTA CORRENTE** (da ottimizzare):
{current_answer}
{ai_overview_section}
**TOP 3 RISPOSTE CONCORRENTI** (fonti citate nell'AI Overview):
{sources_text}

**ENTITÀ/CONCETTI MANCANTI** (presenti nell'AI Overview e competitor):
{entities_text if entities_text else "Nessuna entità mancante significativa"}

**TASK**:
Genera una versione OTTIMIZZATA della risposta corrente che:

1. **PRIORITÀ MASSIMA**: Copre TUTTI i punti chiave dell'AI Overview di Google
2. Mantiene il tono e stile della risposta originale
3. Integra le entità mancanti chiave in modo naturale
4. Risponde in modo DIRETTO e COMPLETO alla query
5. È concisa: massimo {max_words} parole
6. Include informazioni fattuali e specifiche
7. Usa un linguaggio chiaro e accessibile

**IMPORTANTE**:
- L'AI Overview è il benchmark - la tua risposta deve contenere le stesse informazioni chiave
- NON copiare frasi, ma assicurati di coprire gli stessi concetti
- NON aggiungere informazioni non verificabili
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

            word_count = count_words(optimized_answer)
            if word_count > max_words * 1.2:
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

    def optimize_answer(
        self,
        query: str,
        current_answer: str,
        top_sources: List[Dict[str, Any]],
        missing_entities: List[Dict[str, str]],
        max_words: int = 300,
        ai_overview_text: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Alias per generate_optimized_answer"""
        return self.generate_optimized_answer(
            query=query,
            current_answer=current_answer,
            top_sources=top_sources,
            missing_entities=missing_entities,
            max_words=max_words,
            ai_overview_text=ai_overview_text
        )

    def analyze_content_quality(self, text: str) -> Dict[str, Any]:
        """Analizza qualità del contenuto"""
        word_count = count_words(text)
        char_count = len(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        avg_words_per_sentence = word_count / max(sentences, 1)
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
        """Estrae frasi chiave dal testo"""
        if not self.nlp:
            # Fallback semplice: estrai n-grammi
            words = text.lower().split()
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            phrase_counter = Counter(bigrams)
            return [phrase for phrase, _ in phrase_counter.most_common(top_n)]

        doc = self.nlp(text)
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
        phrase_counter = Counter(phrases)
        filtered = [
            phrase for phrase, _ in phrase_counter.most_common(top_n * 2)
            if 2 <= len(phrase.split()) <= 5
        ]
        return filtered[:top_n]

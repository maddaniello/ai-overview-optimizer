"""
Content Analyzer - Business Logic per analisi contenuti
Gestisce:
- Analisi entità con spaCy
- Identificazione gap
- Generazione ottimizzazioni con LLM
"""
from typing import List, Dict, Any, Optional
from collections import Counter
from openai import OpenAI
from utils.logger import get_logger
from utils.helpers import truncate_text, count_words
from config import openai_config

logger = get_logger(__name__)

# Import opzionale di spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✓ spaCy disponibile")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("✗ spaCy non disponibile - analisi entità limitata")


class ContentAnalyzer:
    """Analizzatore di contenuti per ottimizzazione AI Overview"""
    
    def __init__(self, language: str = "it"):
        """
        Args:
            language: Codice lingua (it, en, etc.)
        """
        self.language = language
        self.nlp = None
        
        # Carica modello spaCy se disponibile
        if SPACY_AVAILABLE:
            self._load_spacy_model()
        
        # Inizializza OpenAI client
        if openai_config.api_key:
            self.openai_client = OpenAI(api_key=openai_config.api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key non configurata - funzionalità LLM disabilitate")
    
    def _load_spacy_model(self):
        """Carica modello spaCy per la lingua specificata"""
        models = {
            "it": "it_core_news_lg",
            "en": "en_core_web_lg",
            "de": "de_core_news_lg",
            "fr": "fr_core_news_lg",
            "es": "es_core_news_lg",
        }
        
        model_name = models.get(self.language, "en_core_web_lg")
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✓ Modello spaCy caricato: {model_name}")
        except Exception as e:
            logger.error(f"✗ Errore caricamento modello spaCy: {str(e)}")
            logger.warning(f"Installa con: python -m spacy download {model_name}")
            self.nlp = None
    
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
        comparison_texts: List[str],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Analizza gap di entità tra target e testi di confronto
        
        Args:
            target_text: Testo da ottimizzare
            comparison_texts: Testi competitor
            top_n: Numero di entità top da restituire
            
        Returns:
            Dict con analisi gap
        """
        logger.info("🔍 Analisi gap entità")
        
        # Estrai entità dal target
        target_entities = self.extract_entities(target_text)
        target_entity_texts = set(ent["text"].lower() for ent in target_entities)
        
        logger.info(f"Target: {len(target_entity_texts)} entità uniche")
        
        # Estrai entità dai competitor
        all_competitor_entities = []
        for comp_text in comparison_texts:
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
        
        logger.info(f"✓ Trovate {len(missing_entities)} entità mancanti significative")
        
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
    ) -> Optional[str]:
        """
        Genera versione ottimizzata della risposta usando LLM
        
        Args:
            query: Query originale
            current_answer: Risposta corrente
            top_sources: Top risposte competitor
            missing_entities: Entità mancanti
            max_words: Massimo parole
            
        Returns:
            Risposta ottimizzata o None se LLM non disponibile
        """
        if not self.openai_client:
            logger.warning("OpenAI non disponibile - skip ottimizzazione LLM")
            return None
        
        logger.info("🤖 Generazione risposta ottimizzata con LLM")
        
        # Formatta top sources
        sources_text = "\n\n".join([
            f"**Fonte {i+1}** (Score: {src.get('relevance_score', 0):.2f}):\n{truncate_text(src.get('answer', ''), 200)}"
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

1. ✅ Mantiene il tono e stile della risposta originale
2. ✅ Integra le entità mancanti chiave in modo naturale
3. ✅ Copre tutti gli aspetti rilevanti delle risposte competitor
4. ✅ Risponde in modo DIRETTO e COMPLETO alla query
5. ✅ È concisa: massimo {max_words} parole
6. ✅ Include informazioni fattuali e specifiche
7. ✅ Usa un linguaggio chiaro e accessibile

**IMPORTANTE**:
- NON copiare frasi dai competitor
- NON aggiungere informazioni non verificabili
- Mantieni lo stesso livello di autorevolezza
- Rispondi nella stessa lingua della query
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=openai_config.chat_model,
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
            
            logger.info(f"✓ Risposta ottimizzata generata: {word_count} parole")
            
            return optimized_answer
            
        except Exception as e:
            logger.error(f"✗ Errore generazione ottimizzazione: {str(e)}")
            return None
    
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
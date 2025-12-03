"""
Web Scraper usando Crawl4AI con fallback a BeautifulSoup
"""
import asyncio
from typing import Dict, Optional, List
from bs4 import BeautifulSoup
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from utils.logger import get_logger
from utils.helpers import clean_text, truncate_text

logger = get_logger(__name__)

# Import opzionale di Crawl4AI
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import CrawlerRunConfig
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    CRAWL4AI_AVAILABLE = True
    logger.info("✓ Crawl4AI disponibile")
except ImportError:
    CRAWL4AI_AVAILABLE = False
    logger.warning("✗ Crawl4AI non disponibile, uso solo BeautifulSoup")


class WebScraper:
    """Scraper per estrarre contenuti da pagine web"""
    
    def __init__(self, use_crawl4ai: bool = True):
        """
        Args:
            use_crawl4ai: Se True tenta di usare Crawl4AI, altrimenti usa solo BS4
        """
        self.use_crawl4ai = use_crawl4ai and CRAWL4AI_AVAILABLE
        
        if self.use_crawl4ai:
            logger.info("Scraper inizializzato con Crawl4AI")
        else:
            logger.info("Scraper inizializzato con BeautifulSoup")
    
    async def scrape_url(
        self,
        url: str,
        extract_answer_for_query: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Scrape contenuto da URL
        
        Args:
            url: URL da scrapare
            extract_answer_for_query: Se fornito, estrae risposta specifica alla query
            
        Returns:
            Dict con contenuto estratto
        """
        logger.info(f"🕷️ Scraping: {url}")
        
        try:
            if self.use_crawl4ai:
                result = await self._scrape_with_crawl4ai(url, extract_answer_for_query)
            else:
                result = self._scrape_with_beautifulsoup(url)
            
            logger.info(f"✓ Scraping completato: {len(result.get('content', ''))} caratteri")
            return result
            
        except Exception as e:
            logger.error(f"✗ Errore scraping {url}: {str(e)}")
            
            # Fallback a BeautifulSoup se Crawl4AI fallisce
            if self.use_crawl4ai:
                logger.warning("Tentativo fallback con BeautifulSoup")
                try:
                    return self._scrape_with_beautifulsoup(url)
                except Exception as e2:
                    logger.error(f"✗ Anche fallback fallito: {str(e2)}")
            
            return {
                "url": url,
                "content": "",
                "answer_text": "",
                "error": str(e)
            }
    
    async def scrape_multiple_urls(
        self,
        urls: List[str],
        extract_answer_for_query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Scrape multipli URL in parallelo
        
        Args:
            urls: Lista di URL da scrapare
            extract_answer_for_query: Query per estrazione risposta
            
        Returns:
            Lista di risultati
        """
        logger.info(f"🕷️ Scraping {len(urls)} URL in parallelo")
        
        tasks = [
            self.scrape_url(url, extract_answer_for_query)
            for url in urls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtra errori
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"✗ Errore URL {urls[i]}: {str(result)}")
            else:
                valid_results.append(result)
        
        logger.info(f"✓ Completati {len(valid_results)}/{len(urls)} scraping")
        return valid_results
    
    async def _scrape_with_crawl4ai(
        self,
        url: str,
        query: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Scrape con Crawl4AI (metodo avanzato)
        
        Args:
            url: URL da scrapare
            query: Query per estrazione risposta specifica
            
        Returns:
            Dict con contenuto estratto
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError("Crawl4AI non disponibile")
        
        config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(),
            word_count_threshold=10,
            excluded_tags=['nav', 'footer', 'script', 'style', 'aside'],
            remove_overlay_elements=True,
        )
        
        # Se abbiamo una query, usa LLM extraction
        if query:
            from config import openai_config
            if openai_config.api_key:
                config.extraction_strategy = LLMExtractionStrategy(
                    provider="openai/gpt-4o-mini",
                    api_token=openai_config.api_key,
                    instruction=f"""Estrai SOLO la risposta diretta alla query: "{query}"
                    
Regole:
- Massimo 300 parole
- Solo la parte che risponde alla domanda
- Mantieni entità chiave (nomi, date, numeri)
- Rispondi in italiano se la pagina è in italiano
"""
                )
        
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(
                url=url,
                config=config,
                bypass_cache=True
            )
            
            content = result.markdown if result.success else ""
            answer_text = ""
            
            # Se abbiamo usato LLM extraction
            if hasattr(result, 'extracted_content') and result.extracted_content:
                answer_text = result.extracted_content
            
            return {
                "url": url,
                "content": content,
                "answer_text": answer_text,
                "success": result.success
            }
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def _scrape_with_beautifulsoup(self, url: str) -> Dict[str, str]:
        """
        Scrape con BeautifulSoup (metodo fallback)
        
        Args:
            url: URL da scrapare
            
        Returns:
            Dict con contenuto estratto
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Rimuovi elementi non necessari
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            tag.decompose()
        
        # Estrai testo
        text = soup.get_text(separator=' ', strip=True)
        
        # Pulisci
        text = clean_text(text)
        
        return {
            "url": url,
            "content": text,
            "answer_text": "",  # BS4 non estrae risposte specifiche
            "success": True
        }
    
    def extract_answer_from_content(
        self,
        content: str,
        query: str,
        max_words: int = 300
    ) -> str:
        """
        Estrae risposta da contenuto usando metodo rule-based semplice
        
        Args:
            content: Contenuto completo
            query: Query di riferimento
            max_words: Massimo parole nella risposta
            
        Returns:
            Risposta estratta
        """
        # Split in paragrafi
        paragraphs = [p.strip() for p in content.split('\n') if len(p.strip()) > 50]
        
        # Query keywords
        query_words = set(query.lower().split())
        
        # Trova paragrafi più rilevanti
        scored_paragraphs = []
        for para in paragraphs[:50]:  # Analizza primi 50 paragrafi
            para_lower = para.lower()
            # Conta quante keyword della query sono presenti
            score = sum(1 for word in query_words if word in para_lower)
            if score > 0:
                scored_paragraphs.append((score, para))
        
        # Ordina per score
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])
        
        # Prendi top 3 paragrafi
        top_paragraphs = [p[1] for p in scored_paragraphs[:3]]
        
        # Unisci e tronca
        answer = " ".join(top_paragraphs)
        
        # Tronca a max_words parole
        words = answer.split()
        if len(words) > max_words:
            answer = " ".join(words[:max_words]) + "..."
        
        return answer

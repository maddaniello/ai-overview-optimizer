"""
Content Scraper - Solo BeautifulSoup (Streamlit Cloud compatible)
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, List
from utils.logger import logger
from config import SCRAPING_TIMEOUT, USER_AGENT
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ContentScraper:
    """Scraper per estrazione contenuti web"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = SCRAPING_TIMEOUT
        
        logger.info("ContentScraper inizializzato (BeautifulSoup mode)")
    
    def scrape_page(self, url: str) -> Dict[str, any]:
        """
        Scrape singola pagina
        
        Args:
            url: URL da scrapare
            
        Returns:
            Dict con contenuto estratto
        """
        try:
            logger.info(f"Scraping: {url}")
            
            # HTTP Request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Rimuovi elementi non utili
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                tag.decompose()
            
            # Estrai contenuto
            content = self._extract_content(soup)
            
            logger.success(f"Pagina scraped: {len(content)} caratteri")
            
            return {
                "url": url,
                "content": content,
                "success": True
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore HTTP per {url}: {e}")
            return {"url": url, "content": "", "success": False, "error": str(e)}
        
        except Exception as e:
            logger.error(f"Errore scraping {url}: {e}")
            return {"url": url, "content": "", "success": False, "error": str(e)}
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Estrae contenuto testuale dalla pagina
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Testo estratto
        """
        # Priorità elementi
        priority_tags = ['article', 'main', 'section']
        
        # Cerca contenuto principale
        main_content = None
        for tag in priority_tags:
            main_content = soup.find(tag)
            if main_content:
                break
        
        # Fallback: usa body
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return soup.get_text(separator=' ', strip=True)
        
        # Estrai paragrafi
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
        
        # Pulisci e unisci
        texts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 20:  # Ignora paragrafi troppo corti
                texts.append(text)
        
        return ' '.join(texts)
    
    def extract_answer(self, url: str, query: str) -> str:
        """
        Estrae risposta alla query dalla pagina
        
        Args:
            url: URL pagina
            query: Query di ricerca
            
        Returns:
            Risposta estratta (max 300 parole)
        """
        result = self.scrape_page(url)
        
        if not result["success"]:
            return ""
        
        content = result["content"]
        
        # Semplice: prendi primi 300 parole del contenuto
        words = content.split()[:300]
        answer = ' '.join(words)
        
        logger.info(f"Risposta estratta: {len(words)} parole")
        
        return answer
    
    async def scrape_multiple_async(self, urls: List[str], query: str) -> List[Dict[str, str]]:
        """
        Scrape multiplo URL in parallelo
        
        Args:
            urls: Lista URL
            query: Query per extraction
            
        Returns:
            Lista dict con risposte
        """
        logger.info(f"Scraping {len(urls)} URLs in parallelo")
        
        # Usa ThreadPoolExecutor per parallelizzare requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            loop = asyncio.get_event_loop()
            
            # Crea tasks
            tasks = [
                loop.run_in_executor(executor, self.extract_answer, url, query)
                for url in urls
            ]
            
            # Attendi risultati
            answers = await asyncio.gather(*tasks)
        
        # Formatta risultati
        results = []
        for url, answer in zip(urls, answers):
            if answer:
                results.append({
                    "url": url,
                    "answer": answer,
                    "source": "scraped"
                })
        
        logger.success(f"{len(results)}/{len(urls)} pagine scraped con successo")
        
        return results

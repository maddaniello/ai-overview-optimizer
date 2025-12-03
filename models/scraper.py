"""
Content Scraper - BeautifulSoup (Streamlit Cloud compatible)
Con headers realistici e retry per evitare 403
"""
import requests
import random
import time
from bs4 import BeautifulSoup
from typing import Dict, Optional, List, Any
from utils.logger import logger
from config import SCRAPING_TIMEOUT
import asyncio
from concurrent.futures import ThreadPoolExecutor

# User agents realistici (Chrome/Firefox recenti)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]


class ContentScraper:
    """Scraper per estrazione contenuti web con anti-403"""

    def __init__(self):
        self.timeout = SCRAPING_TIMEOUT
        logger.info("ContentScraper inizializzato (BeautifulSoup mode)")

    def _get_session(self) -> requests.Session:
        """Crea session con headers realistici"""
        session = requests.Session()

        # Headers che simulano un browser reale
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        session.headers.update(headers)
        return session

    def scrape_page(self, url: str, retries: int = 2) -> Dict[str, Any]:
        """
        Scrape singola pagina con retry

        Args:
            url: URL da scrapare
            retries: Numero di retry in caso di fallimento

        Returns:
            Dict con contenuto estratto
        """
        last_error = None

        for attempt in range(retries + 1):
            try:
                logger.info(f"Scraping: {url} (tentativo {attempt + 1})")

                # Nuova session per ogni tentativo (headers diversi)
                session = self._get_session()

                # Piccolo delay random per sembrare più umano
                if attempt > 0:
                    time.sleep(random.uniform(0.5, 1.5))

                # HTTP Request
                response = session.get(url, timeout=self.timeout, allow_redirects=True)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.content, 'lxml')

                # Rimuovi elementi non utili
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
                    tag.decompose()

                # Estrai contenuto
                content = self._extract_content(soup)

                if content and len(content) > 50:
                    logger.success(f"Pagina scraped: {len(content)} caratteri")
                    return {
                        "url": url,
                        "content": content,
                        "success": True
                    }
                else:
                    logger.warning(f"Contenuto troppo corto per {url}")
                    last_error = "Contenuto vuoto o troppo corto"
                    continue

            except requests.exceptions.HTTPError as e:
                last_error = str(e)
                if e.response is not None and e.response.status_code == 403:
                    logger.warning(f"403 Forbidden per {url} - tentativo {attempt + 1}")
                    continue
                else:
                    logger.error(f"HTTP Error {e.response.status_code if e.response else 'N/A'} per {url}")
                    break

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.error(f"Errore request per {url}: {e}")
                break

            except Exception as e:
                last_error = str(e)
                logger.error(f"Errore scraping {url}: {e}")
                break

        # Tutti i tentativi falliti
        return {"url": url, "content": "", "success": False, "error": last_error or "Unknown error"}

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Estrae contenuto testuale dalla pagina

        Args:
            soup: BeautifulSoup object

        Returns:
            Testo estratto
        """
        # Priorità elementi
        priority_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.content',
            '#content',
            'section'
        ]

        # Cerca contenuto principale
        main_content = None
        for selector in priority_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # Fallback: usa body
        if not main_content:
            main_content = soup.find('body')

        if not main_content:
            return soup.get_text(separator=' ', strip=True)

        # Estrai paragrafi
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'td', 'span'])

        # Pulisci e unisci
        texts = []
        seen = set()  # Evita duplicati

        for p in paragraphs:
            text = p.get_text(strip=True)
            # Ignora testi troppo corti o duplicati
            if len(text) > 20 and text not in seen:
                texts.append(text)
                seen.add(text)

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
            logger.warning(f"Scraping fallito per {url}: {result.get('error', 'N/A')}")
            return ""

        content = result["content"]

        # Prendi primi 300 parole del contenuto
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
        with ThreadPoolExecutor(max_workers=3) as executor:  # Max 3 per evitare rate limiting
            loop = asyncio.get_event_loop()

            # Crea tasks
            tasks = [
                loop.run_in_executor(executor, self.extract_answer, url, query)
                for url in urls
            ]

            # Attendi risultati
            answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Formatta risultati
        results = []
        for url, answer in zip(urls, answers):
            if isinstance(answer, Exception):
                logger.error(f"Eccezione per {url}: {answer}")
                continue
            if answer:
                results.append({
                    "url": url,
                    "answer": answer,
                    "source": "scraped"
                })

        success_count = len(results)
        logger.success(f"{success_count}/{len(urls)} pagine scraped con successo")

        return results

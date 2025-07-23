import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, TypedDict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Article(TypedDict):
    title: str
    url: str
    summary: str
    published: Optional[str]
    source: str


class RSSCollector:
    def __init__(self, sources: List[str]):
        self.sources = sources

    def fetch_feeds(self) -> List[Article]:
        articles: List[Article] = []

        for source_url in self.sources:
            try:
                logger.info(f"Lecture du flux RSS : {source_url}")
                feed = feedparser.parse(source_url)

                for entry in feed.entries:
                    # Vérification champs essentiels
                    if not all(hasattr(entry, attr) for attr in ("title", "link", "summary")):
                        continue

                    title = entry.title.strip()
                    url = entry.link.strip()
                    summary = self._clean_html(entry.summary)
                    
                    # Prioriser published_parsed si disponible, sinon published
                    published_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published_date = datetime(*entry.published_parsed[:6]).isoformat()
                        except (TypeError, ValueError):
                            pass
                    
                    if not published_date:
                        published_date = self._parse_date(entry.get("published"))

                    articles.append({
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "published": published_date,
                        "source": source_url
                    })

            except Exception as e:
                logger.warning(f"Erreur lors de la récupération de {source_url} : {e}")

        return articles

    def _clean_html(self, html_text: str) -> str:
        """Nettoie le HTML pour obtenir un texte brut propre."""
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    def _parse_date(self, raw_date: Optional[str]) -> Optional[str]:
        """Convertit une date brute RSS en format standard ISO (YYYY-MM-DD)."""
        if not raw_date:
            return None
            
        try:
            # Utiliser time.struct_time si disponible dans l'entry
            import time
            from email.utils import parsedate_tz
            
            # Essayer d'abord parsedate_tz pour les dates RFC 2822
            parsed_tuple = parsedate_tz(raw_date)
            if parsed_tuple:
                # Convertir en timestamp puis datetime
                timestamp = time.mktime(parsed_tuple[:9])
                dt = datetime.fromtimestamp(timestamp)
                return dt.isoformat()
            
            # Fallback: parser direct feedparser
            parsed = feedparser._parse_date(raw_date) if hasattr(feedparser, '_parse_date') else None
            if parsed:
                return datetime(*parsed[:6]).isoformat()
                
        except Exception as e:
            logger.debug(f"Erreur parsing date '{raw_date}': {e}")
            
        return None

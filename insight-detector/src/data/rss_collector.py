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
                    published = self._parse_date(entry.get("published"))

                    articles.append({
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "published": published,
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
        try:
            if raw_date:
                parsed = feedparser._parse_date(raw_date)
                if parsed:
                    return datetime(*parsed[:6]).isoformat()
        except Exception:
            pass
        return None

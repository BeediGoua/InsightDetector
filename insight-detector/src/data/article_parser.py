from newspaper import Article
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ArticleParser:
    def __init__(self, language: str = 'fr'):
        self.language = language

    def extract_text(self, url: str) -> Optional[str]:
        try:
            logger.info(f"Téléchargement de l'article : {url}")
            article = Article(url, language=self.language)
            article.download()
            article.parse()
            text = article.text.strip()

            # Filtrage simple : longueur minimale
            if len(text) < 300:
                logger.warning(f"Article trop court ou vide : {url}")
                return None

            return text

        except Exception as e:
            logger.error(f"Échec parsing {url} : {e}")
            return None

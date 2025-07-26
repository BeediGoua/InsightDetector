# src/data/database_manager.py

from sqlalchemy.orm import Session
from datetime import datetime
from dateutil.parser import parse  
from data.models import Article

class DatabaseManager:
    def __init__(self, session: Session):
        self.session = session

    def get_all_articles(self):
        """Retourne tous les articles sous forme de dictionnaires."""
        return [article.to_dict() for article in self.session.query(Article).all()]

    def get_latest_articles(self, limit: int = 10):
        """Retourne les derniers articles (objets SQLAlchemy)."""
        return (
            self.session.query(Article)
            .order_by(Article.created_at.desc())
            .limit(limit)
            .all()
        )

    def create_article(self, data: dict) -> int:
        """
        Crée un nouvel article (si non existant via l'URL), et retourne son ID.
        Les champs 'published' et 'created_at' sont convertis en datetime si nécessaire.
        """
        # Vérifier doublon via l'URL
        existing = self.session.query(Article).filter_by(url=data["url"]).first()
        if existing:
            return existing.id

        #  Conversion robuste des dates
        def to_datetime(value, fallback=None):
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                try:
                    return parse(value)
                except Exception:
                    pass
            return fallback

        published = to_datetime(data.get("published"))
        created_at = to_datetime(data.get("created_at"), fallback=datetime.utcnow())

        # Création de l'article
        article = Article(
            title=data["title"],
            url=data["url"],
            summary=data.get("summary"),
            text=data.get("text"),
            published=published,
            source=data.get("source", "unknown"),
            created_at=created_at
        )

        try:
            self.session.add(article)
            self.session.commit()
            return article.id
        except Exception as e:
            self.session.rollback()
            raise e

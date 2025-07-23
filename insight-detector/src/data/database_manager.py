from sqlalchemy.orm import Session
from datetime import datetime
from data.models import Article

class DatabaseManager:
    def __init__(self, session: Session):
        self.session = session

    def create_article(self, data: dict) -> int:
        # Vérifie si l’article existe déjà
        existing = self.session.query(Article).filter_by(url=data["url"]).first()
        if existing:
            return existing.id

        article = Article(
            title=data["title"],
            url=data["url"],
            summary=data.get("summary"),
            text=data.get("text"),
            published=data.get("published"),
            source=data["source"],
            created_at=datetime.utcnow()
        )
        self.session.add(article)
        self.session.commit()
        return article.id

    def get_latest_articles(self, limit: int = 10):
        return self.session.query(Article).order_by(Article.created_at.desc()).limit(limit).all()

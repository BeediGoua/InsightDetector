# src/data/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    summary = Column(Text)
    text = Column(Text)
    published = Column(DateTime)
    source = Column(String)
    url = Column(String, unique=True)
    created_at = Column(DateTime)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "text": self.text,
            "published": self.published.isoformat() if self.published else None,
            "source": self.source,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

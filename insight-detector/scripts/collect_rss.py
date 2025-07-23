from src.data.rss_collector import RSSCollector
from src.data.article_parser import ArticleParser
from src.config.database import SessionLocal
from src.data.database_manager import DatabaseManager

if __name__ == "__main__":
    sources = [
        "https://www.lemonde.fr/rss/une.xml",
        "https://www.france24.com/fr/rss"
    ]

    collector = RSSCollector(sources)
    parser = ArticleParser()
    session = SessionLocal()
    db = DatabaseManager(session)

    articles = collector.fetch_feeds()

    for a in articles[:10]:
        a["text"] = parser.extract_text(a["url"])  
        article_id = db.create_article(a)
        print(f"Article {article_id} inséré : {a['title']}")

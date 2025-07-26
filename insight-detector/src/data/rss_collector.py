# src/data/rss_collector.py 

import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, TypedDict
from datetime import datetime, timedelta
import logging
from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)

class Article(TypedDict):
    title: str
    url: str
    summary: str
    published: Optional[str]
    source: str

class RSSCollector:
    def __init__(self, sources: List[str], days_back: int = 7, timeout: int = 30):
        """
        Collecteur RSS avec filtrage temporel
        
        Args:
            sources: Liste des URLs RSS
            days_back: Nombre de jours à collecter (défaut: 7)
            timeout: Timeout pour les requêtes
        """
        self.sources = sources
        self.days_back = days_back
        self.timeout = timeout
        
        # Calcul de la date limite (7 jours par défaut)
        self.date_cutoff = datetime.now() - timedelta(days=days_back)
        logger.info(f" Collecte articles depuis: {self.date_cutoff.strftime('%Y-%m-%d %H:%M')}")

    def fetch_feeds(self, filter_by_date: bool = True) -> List[Article]:
        """
        Collecte tous les flux avec filtrage optionnel par date
        
        Args:
            filter_by_date: Si True, filtre les articles selon days_back
        """
        all_articles: List[Article] = []
        stats = {"total": 0, "filtered": 0, "no_date": 0, "errors": 0}

        for source_url in self.sources:
            try:
                logger.info(f" Lecture flux RSS: {source_url}")
                source_articles = self.fetch_single_feed(source_url, filter_by_date, stats)
                all_articles.extend(source_articles)
                
            except Exception as e:
                logger.error(f" Erreur source {source_url}: {e}")
                stats["errors"] += 1

        #  Rapport final
        self._log_collection_stats(stats, len(all_articles))
        return all_articles

    def fetch_single_feed(self, source_url: str, filter_by_date: bool, stats: Dict) -> List[Article]:
        """Collecte d'un flux RSS individuel avec filtrage"""
        articles: List[Article] = []
        
        # Configuration feedparser avec timeout
        feed = feedparser.parse(source_url)
        
        if hasattr(feed, 'status') and feed.status != 200:
            logger.warning(f"  Status HTTP {feed.status} pour {source_url}")
            
        for entry in feed.entries:
            stats["total"] += 1
            
            # Validation champs essentiels
            if not self._validate_entry(entry):
                continue

            # Extraction données de base
            article_data = self._extract_article_data(entry, source_url)
            
            if not article_data:
                continue
                
            #  FILTRAGE PAR DATE
            if filter_by_date:
                published_date = self._parse_article_date(entry)
                
                if published_date is None:
                    stats["no_date"] += 1
                    logger.debug(f" Pas de date pour: {article_data['title'][:50]}...")
                    
                    articles.append(article_data)
                    continue
                
                # Vérification si l'article est dans la période désirée
                if published_date < self.date_cutoff:
                    stats["filtered"] += 1
                    logger.debug(f" Article trop ancien: {published_date} < {self.date_cutoff}")
                    continue
                else:
                    logger.debug(f" Article récent: {published_date}")

            articles.append(article_data)

        logger.info(f" {len(articles)} articles récupérés de {source_url}")
        return articles

    def _validate_entry(self, entry) -> bool:
        """Validation des champs essentiels d'un article RSS"""
        required_fields = ("title", "link")
        missing_fields = [field for field in required_fields if not hasattr(entry, field)]
        
        if missing_fields:
            logger.debug(f"  Champs manquants: {missing_fields}")
            return False
            
        if not entry.title.strip() or not entry.link.strip():
            logger.debug("  Titre ou URL vide")
            return False
            
        return True

    def _extract_article_data(self, entry, source_url: str) -> Optional[Dict]:
        """Extraction des données d'un article RSS"""
        try:
            title = entry.title.strip()
            url = entry.link.strip()
            summary = self._clean_html(getattr(entry, 'summary', ''))
            
            # Date de publication (pour stockage)
            published_date = self._parse_article_date(entry)
            published_iso = published_date.isoformat() if published_date else None

            return {
                "title": title,
                "url": url,
                "summary": summary,
                "published": published_iso,
                "source": source_url
            }
            
        except Exception as e:
            logger.error(f" Erreur extraction article: {e}")
            return None

    def _parse_article_date(self, entry) -> Optional[datetime]:
        """
        Parse la date d'un article RSS de manière robuste
        Priorité: published_parsed > published > updated_parsed > updated
        """
        # Méthode 1: published_parsed (timestamp struct)
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except (TypeError, ValueError) as e:
                logger.debug(f"Erreur published_parsed: {e}")

        # Méthode 2: published (string)
        if hasattr(entry, 'published') and entry.published:
            try:
                return parse_date(entry.published)
            except Exception as e:
                logger.debug(f"Erreur published string: {e}")

        # Méthode 3: updated_parsed (fallback)
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            try:
                return datetime(*entry.updated_parsed[:6])
            except (TypeError, ValueError) as e:
                logger.debug(f"Erreur updated_parsed: {e}")

        # Méthode 4: updated (string fallback)
        if hasattr(entry, 'updated') and entry.updated:
            try:
                return parse_date(entry.updated)
            except Exception as e:
                logger.debug(f"Erreur updated string: {e}")

        # Aucune date trouvée
        return None

    def _clean_html(self, html_text: str) -> str:
        """Nettoie le HTML pour obtenir un texte brut propre"""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(separator=" ")
        return " ".join(text.split())

    def _log_collection_stats(self, stats: Dict, final_count: int):
        """Log des statistiques de collecte"""
        logger.info("="*50)
        logger.info(" STATISTIQUES DE COLLECTE RSS")
        logger.info(f" Articles trouvés total    : {stats['total']}")
        logger.info(f" Articles filtrés (trop anciens): {stats['filtered']}")
        logger.info(f" Articles sans date        : {stats['no_date']}")
        logger.info(f" Sources en erreur         : {stats['errors']}")
        logger.info(f" Articles finaux collectés : {final_count}")
        
        if stats["total"] > 0:
            retention_rate = (final_count / stats["total"]) * 100
            logger.info(f" Taux de rétention         : {retention_rate:.1f}%")
        
        logger.info("="*50)

    # MÉTHODES UTILITAIRES SUPPLÉMENTAIRES
    
    def get_date_range_summary(self) -> Dict:
        """Retourne un résumé de la période de collecte"""
        now = datetime.now()
        return {
            "start_date": self.date_cutoff.isoformat(),
            "end_date": now.isoformat(),
            "days_back": self.days_back,
            "total_hours": int((now - self.date_cutoff).total_seconds() / 3600)
        }

    def fetch_feeds_by_custom_date(self, start_date: datetime, end_date: datetime) -> List[Article]:
        """
        Collecte avec une période personnalisée
        
        Args:
            start_date: Date de début (incluse)
            end_date: Date de fin (incluse)
        """
        original_cutoff = self.date_cutoff
        self.date_cutoff = start_date
        
        try:
            articles = self.fetch_feeds(filter_by_date=True)
            # Filtrage supplémentaire pour la date de fin
            filtered_articles = []
            
            for article in articles:
                if article['published']:
                    try:
                        pub_date = parse_date(article['published'])
                        if start_date <= pub_date <= end_date:
                            filtered_articles.append(article)
                    except:
                        # Inclure les articles avec des dates non-parsables
                        filtered_articles.append(article)
                else:
                    # Inclure les articles sans date
                    filtered_articles.append(article)
                    
            logger.info(f" Période personnalisée: {len(filtered_articles)} articles entre {start_date} et {end_date}")
            return filtered_articles
            
        finally:
            self.date_cutoff = original_cutoff

    def test_single_source(self, source_url: str) -> Dict:
        """
        Test d'une source RSS individuelle avec diagnostics
        Utile pour débugger les sources problématiques
        """
        logger.info(f" TEST SOURCE: {source_url}")
        
        try:
            feed = feedparser.parse(source_url)
            
            diagnostics = {
                "url": source_url,
                "status": getattr(feed, 'status', 'unknown'),
                "total_entries": len(feed.entries),
                "feed_title": getattr(feed.feed, 'title', 'N/A'),
                "feed_updated": getattr(feed.feed, 'updated', 'N/A'),
                "articles_with_dates": 0,
                "date_range": {"oldest": None, "newest": None},
                "sample_articles": []
            }
            
            # Analyse des articles
            dates = []
            for i, entry in enumerate(feed.entries[:5]):  # Échantillon des 5 premiers
                pub_date = self._parse_article_date(entry)
                if pub_date:
                    dates.append(pub_date)
                    diagnostics["articles_with_dates"] += 1
                
                diagnostics["sample_articles"].append({
                    "title": getattr(entry, 'title', 'N/A')[:100],
                    "published": pub_date.isoformat() if pub_date else None,
                    "has_summary": bool(getattr(entry, 'summary', False))
                })
            
            # Analyse des dates
            if dates:
                diagnostics["date_range"] = {
                    "oldest": min(dates).isoformat(),
                    "newest": max(dates).isoformat()
                }
            
            logger.info(f" Test terminé: {diagnostics['total_entries']} articles trouvés")
            return diagnostics
            
        except Exception as e:
            logger.error(f" Erreur test source: {e}")
            return {"url": source_url, "error": str(e)}
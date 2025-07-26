# src/export/structured_exporter.py

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class StructuredExporter:
    """
    Exporteur pour sauvegarder les articles enrichis dans différents formats
    """
    
    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def export(self, articles: List[Dict], format: str = "json") -> bool:
        """
        Exporte les articles dans le format spécifié
        
        Args:
            articles: Liste des articles enrichis
            format: "json", "csv", "excel" ou "all"
        """
        try:
            if format == "json" or format == "all":
                self._export_json(articles)
                
            if format == "csv" or format == "all":
                self._export_csv(articles)
                
            if format == "excel" or format == "all":
                self._export_excel(articles)
                
            logger.info(f" Export terminé : {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f" Erreur export : {e}")
            return False
    
    def _export_json(self, articles: List[Dict]):
        """Export au format JSON"""
        output_file = self.output_path.with_suffix('.json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        logger.info(f" JSON exporté : {output_file}")
        
    def _export_csv(self, articles: List[Dict]):
        """Export au format CSV (sans embeddings pour la lisibilité)"""
        output_file = self.output_path.with_suffix('.csv')
        
        # Préparer les données pour CSV (exclure embeddings)
        csv_articles = []
        for article in articles:
            csv_article = {k: v for k, v in article.items() if k != 'embedding'}
            
            # Convertir entities en string JSON si présent
            if 'entities' in csv_article and csv_article['entities']:
                csv_article['entities'] = json.dumps(csv_article['entities'])
                
            csv_articles.append(csv_article)
        
        df = pd.DataFrame(csv_articles)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f" CSV exporté : {output_file}")
        
    def _export_excel(self, articles: List[Dict]):
        """Export au format Excel avec plusieurs onglets"""
        output_file = self.output_path.with_suffix('.xlsx')
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Onglet 1: Articles complets (sans embeddings)
            main_articles = []
            for article in articles:
                main_article = {k: v for k, v in article.items() if k != 'embedding'}
                if 'entities' in main_article and main_article['entities']:
                    main_article['entities'] = json.dumps(main_article['entities'])
                main_articles.append(main_article)
                
            df_main = pd.DataFrame(main_articles)
            df_main.to_excel(writer, sheet_name='Articles', index=False)
            
            # Onglet 2: Statistiques qualité
            if 'quality_score' in df_main.columns:
                quality_stats = df_main['quality_score'].describe()
                quality_df = pd.DataFrame(quality_stats).reset_index()
                quality_df.columns = ['Statistique', 'Valeur']
                quality_df.to_excel(writer, sheet_name='Qualité', index=False)
            
            # Onglet 3: Répartition par source
            if 'source' in df_main.columns:
                source_counts = df_main['source'].value_counts().reset_index()
                source_counts.columns = ['Source', 'Nombre_articles']
                source_counts.to_excel(writer, sheet_name='Sources', index=False)
                
        logger.info(f" Excel exporté : {output_file}")

    def export_light(self, articles: List[Dict]) -> Path:
        """
        Exporte une version légère (sans embeddings) pour partage/analyse rapide
        """
        light_articles = [
            {
                "id": art.get("id"),
                "title": art.get("title"),
                "summary": art.get("summary"),
                "source": art.get("source"),
                "published": art.get("published"),
                "language": art.get("language"),
                "quality_score": art.get("quality_score"),
                "entities": art.get("entities")
            }
            for art in articles
        ]
        
        light_path = self.output_path.parent / f"{self.output_path.stem}_light.json"
        
        with open(light_path, 'w', encoding='utf-8') as f:
            json.dump(light_articles, f, ensure_ascii=False, indent=2)
            
        logger.info(f" Version légère exportée : {light_path}")
        return light_path

    def export_embeddings_only(self, articles: List[Dict]) -> Optional[Path]:
        """
        Exporte uniquement les embeddings pour analyse ML
        """
        embeddings_data = []
        
        for article in articles:
            if article.get('embedding'):
                embeddings_data.append({
                    'id': article.get('id'),
                    'title': article.get('title'),
                    'embedding': article['embedding']
                })
                
        if not embeddings_data:
            logger.warning("  Aucun embedding trouvé pour export")
            return None
            
        embeddings_path = self.output_path.parent / f"{self.output_path.stem}_embeddings.json"
        
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f" Embeddings exportés : {embeddings_path}")
        return embeddings_path
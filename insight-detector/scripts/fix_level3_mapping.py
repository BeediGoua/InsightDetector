#!/usr/bin/env python3
"""
Script pour corriger le mapping Level3 et améliorer la correspondance articles → source_id
"""

import sys
import json
import hashlib
import pandas as pd
from pathlib import Path

def find_project_root():
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / "src").exists() and (parent / "outputs").exists():
            return parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
outputs_dir = PROJECT_ROOT / "outputs"
data_dir = PROJECT_ROOT / "data"

def create_source_id(url: str) -> str:
    """Crée un source_id cohérent avec le mapping unifié"""
    if not url or url == "nan":
        return None
    normalized_url = url.strip().lower()
    return hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()[:16]

def main():
    print("Correction du mapping Level3...")
    
    # 1. Charger le mapping unifié existant
    unified_mapping = pd.read_csv(outputs_dir / "unified_mapping_complete.csv")
    
    # 2. Charger les articles sources
    articles_file = data_dir / "exports" / "raw_articles.json"
    with open(articles_file, "r", encoding="utf-8") as f:
        articles_data = json.load(f)
    
    articles_df = pd.DataFrame(articles_data)
    
    # 3. Créer un mapping amélioré source_id → article avec texte complet
    enhanced_mapping = []
    
    for _, row in unified_mapping.iterrows():
        source_id = row['source_id']
        level2_id = row['level2_id']
        strategy = row['strategy']
        
        # Trouver l'article correspondant par URL
        url = row.get('url', '')
        if url and url != 'nan':
            # Chercher l'article avec cette URL
            matching_articles = articles_df[articles_df['url'].str.lower() == url.lower()]
            
            if len(matching_articles) > 0:
                article = matching_articles.iloc[0]
                text = article.get('text', '')
                has_text = isinstance(text, str) and len(text.strip()) > 0
                enough_length = has_text and len(text) >= 300  # Seuil abaissé
                
                enhanced_mapping.append({
                    'level2_id': level2_id,
                    'source_id': source_id,
                    'article_id': article['id'],
                    'url': url,
                    'title': article.get('title', ''),
                    'text': text,
                    'strategy': strategy,
                    'has_text': has_text,
                    'enough_length': enough_length,
                    'text_length': len(text) if has_text else 0
                })
    
    enhanced_df = pd.DataFrame(enhanced_mapping)
    
    # 4. Statistiques d'amélioration
    total_entries = len(enhanced_df)
    with_text = enhanced_df['has_text'].sum()
    enough_length = enhanced_df['enough_length'].sum()
    
    print(f"Mapping ameliore cree:")
    print(f"   - Total entries: {total_entries}")
    print(f"   - Avec texte: {with_text} ({with_text/total_entries*100:.1f}%)")
    print(f"   - Texte suffisant: {enough_length} ({enough_length/total_entries*100:.1f}%)")
    
    # 5. Sauvegarder le mapping amélioré
    enhanced_df.to_csv(outputs_dir / "level3_enhanced_mapping.csv", index=False)
    
    # 6. Créer un mapping simplifié pour Level3
    level3_mapping = enhanced_df[enhanced_df['has_text']][
        ['level2_id', 'source_id', 'text', 'strategy', 'has_text', 'enough_length', 'text_length']
    ].copy()
    
    level3_mapping.to_csv(outputs_dir / "level3_text_mapping.csv", index=False)
    
    print(f"Fichiers crees:")
    print(f"   - {outputs_dir / 'level3_enhanced_mapping.csv'}")
    print(f"   - {outputs_dir / 'level3_text_mapping.csv'}")
    
    # 7. Analyser les articles sans texte pour diagnostic
    no_text = enhanced_df[~enhanced_df['has_text']]
    if len(no_text) > 0:
        print(f"\n{len(no_text)} entrees sans texte:")
        sample = no_text.head(5)
        for _, row in sample.iterrows():
            print(f"   - {row['level2_id']}: {row['title'][:50]}...")
    
    return enhanced_df

if __name__ == "__main__":
    enhanced_mapping = main()
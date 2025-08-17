#!/usr/bin/env python3
"""
Script pour réparer les données Level 1 en ajoutant les textes manquants.
Fusionne level1_heuristic_enriched_results.csv avec all_summaries_and_scores.json
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_level1_data():
    """Répare les données Level 1 en ajoutant la colonne 'text' manquante."""
    
    # Chemins des fichiers
    project_root = Path(__file__).parent.parent.parent
    
    # Données Level 1 (sans textes)
    level1_file = project_root / 'data' / 'detection' / 'level1_heuristic_enriched_results.csv'
    
    # Données sources (avec textes)
    source_file = project_root / 'data' / 'results' / 'archive_20250803_133252' / 'all_summaries_and_scores.json'
    
    # Fichier de sortie corrigé
    output_file = project_root / 'data' / 'detection' / 'level1_with_texts.csv'
    
    logger.info("Chargement des données Level 1...")
    df_level1 = pd.read_csv(level1_file)
    logger.info(f"Level 1: {len(df_level1)} résumés chargés")
    
    logger.info("Chargement des textes sources...")
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    summaries = source_data.get('summaries', [])
    logger.info(f"Sources: {len(summaries)} résumés avec textes")
    
    # Création du mapping ID -> texte
    text_mapping = {}
    for i, summary in enumerate(summaries):
        # ID basé sur l'index (comme dans Level 1)
        summary_id = f"{i}_adaptive"  # Format utilisé par Level 1
        
        # Récupération du texte du résumé
        ensemble_summary = summary.get('ensemble_summary', {})
        text = ensemble_summary.get('summary', '')
        
        if not text and 'individual_summaries' in summary:
            # Fallback sur le premier résumé individuel
            individual = summary['individual_summaries'][0]
            text = individual.get('text', individual.get('summary', ''))
        
        text_mapping[summary_id] = text
    
    logger.info(f"Mapping créé: {len(text_mapping)} textes")
    
    # Ajout de la colonne text au DataFrame Level 1
    df_level1['text'] = df_level1['id'].map(text_mapping)
    
    # Vérification des résultats
    missing_texts = df_level1['text'].isna().sum()
    if missing_texts > 0:
        logger.warning(f"{missing_texts} textes manquants après fusion")
        # Remplissage des textes manquants
        df_level1['text'] = df_level1['text'].fillna("Texte manquant - nécessite investigation")
    
    # Sauvegarde
    df_level1.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Données corrigées sauvées: {output_file}")
    
    # Statistiques finales
    avg_length = df_level1['text'].str.len().mean()
    logger.info(f"Longueur moyenne des textes: {avg_length:.1f} caractères")
    logger.info(f"Textes valides: {(~df_level1['text'].isna()).sum()}/{len(df_level1)}")
    
    return output_file

if __name__ == "__main__":
    fix_level1_data()
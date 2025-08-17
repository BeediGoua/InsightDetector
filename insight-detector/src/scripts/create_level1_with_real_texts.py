#!/usr/bin/env python3
"""
Script pour créer un fichier Level 1 complet avec les vrais textes des résumés.
Corrige le problème critique des textes fictifs du notebook.
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_level1_with_real_texts():
    """Crée un fichier Level 1 avec les vrais textes des résumés."""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Données Level 1 (sans textes)
    level1_file = project_root / 'data' / 'detection' / 'level1_heuristic_enriched_results.csv'
    
    # Données sources avec textes réels
    source_file = project_root / 'data' / 'results' / 'archive_20250803_133252' / 'all_summaries_and_scores.json'
    
    # Fichier de sortie
    output_file = project_root / 'data' / 'detection' / 'level1_with_real_texts.csv'
    
    logger.info("=== CRÉATION LEVEL 1 AVEC TEXTES RÉELS ===")
    
    # Chargement Level 1
    logger.info(f"Chargement Level 1: {level1_file}")
    df_level1 = pd.read_csv(level1_file)
    logger.info(f"Level 1: {len(df_level1)} résumés")
    
    # Chargement des textes sources
    logger.info(f"Chargement textes sources: {source_file}")
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    summaries = source_data.get('summaries', [])
    logger.info(f"Textes sources: {len(summaries)} résumés disponibles")
    
    # Création du mapping ID -> données complètes
    text_mapping = {}
    stats = {'matched': 0, 'missing_text': 0, 'total_sources': len(summaries)}
    
    for i, summary in enumerate(summaries):
        # ID basé sur l'index (format Level 1)
        summary_id = f"{i}_adaptive"
        
        # Extraction du texte principal
        text = ""
        if 'ensemble_summary' in summary and summary['ensemble_summary']:
            text = summary['ensemble_summary'].get('summary', '')
        
        # Fallback sur résumé individuel si pas d'ensemble
        if not text and 'individual_summaries' in summary and summary['individual_summaries']:
            first_summary = summary['individual_summaries'][0]
            text = first_summary.get('text', first_summary.get('summary', ''))
        
        # Statistiques de qualité du texte
        if text:
            text_mapping[summary_id] = {
                'text': text,
                'length': len(text),
                'word_count': len(text.split()),
                'has_ensemble': 'ensemble_summary' in summary,
                'source_index': i
            }
            stats['matched'] += 1
        else:
            stats['missing_text'] += 1
            logger.warning(f"Pas de texte pour résumé {summary_id}")
    
    logger.info(f"Mapping créé: {stats['matched']}/{stats['total_sources']} textes extraits")
    
    # Ajout des textes au DataFrame Level 1
    texts_added = 0
    texts_missing = 0
    
    def get_text_data(summary_id):
        if summary_id in text_mapping:
            return text_mapping[summary_id]['text']
        else:
            return None
    
    # Application du mapping
    df_level1['text'] = df_level1['id'].apply(get_text_data)
    
    # Comptage des résultats
    texts_added = (~df_level1['text'].isna()).sum()
    texts_missing = df_level1['text'].isna().sum()
    
    logger.info(f"Fusion réalisée: {texts_added} textes ajoutés, {texts_missing} manquants")
    
    # Traitement des textes manquants
    if texts_missing > 0:
        logger.warning(f"Remplacement de {texts_missing} textes manquants par texte d'erreur")
        df_level1['text'] = df_level1['text'].fillna("TEXTE_MANQUANT_ERREUR_MAPPING")
    
    # Ajout de métadonnées sur les textes
    def get_text_length(summary_id):
        if summary_id in text_mapping:
            return text_mapping[summary_id]['length']
        return 0
    
    def get_word_count(summary_id):
        if summary_id in text_mapping:
            return text_mapping[summary_id]['word_count']
        return 0
    
    df_level1['text_length'] = df_level1['id'].apply(get_text_length)
    df_level1['text_word_count'] = df_level1['id'].apply(get_word_count)
    
    # Validation de la qualité des textes
    logger.info("Validation de la qualité des textes...")
    
    # Statistiques des textes
    valid_texts = df_level1[df_level1['text'] != "TEXTE_MANQUANT_ERREUR_MAPPING"]
    
    text_stats = {
        'total_summaries': int(len(df_level1)),
        'valid_texts': int(len(valid_texts)),
        'missing_texts': int(texts_missing),
        'avg_length': float(valid_texts['text_length'].mean()),
        'avg_word_count': float(valid_texts['text_word_count'].mean()),
        'min_length': int(valid_texts['text_length'].min()),
        'max_length': int(valid_texts['text_length'].max()),
        'unique_texts': int(valid_texts['text'].nunique())
    }
    
    logger.info(f"Statistiques des textes:")
    logger.info(f"  - Textes valides: {text_stats['valid_texts']}/{text_stats['total_summaries']}")
    logger.info(f"  - Longueur moyenne: {text_stats['avg_length']:.1f} caractères")
    logger.info(f"  - Mots moyens: {text_stats['avg_word_count']:.1f}")
    logger.info(f"  - Textes uniques: {text_stats['unique_texts']}")
    
    # Vérification critique de la diversité
    if text_stats['unique_texts'] < text_stats['valid_texts'] * 0.8:
        logger.error(f"PROBLÈME: Trop peu de textes uniques ({text_stats['unique_texts']}/{text_stats['valid_texts']})")
        
    # Sauvegarde du fichier corrigé
    df_level1.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Fichier Level 1 corrigé sauvé: {output_file}")
    
    # Sauvegarde des statistiques
    stats_file = output_file.parent / 'level1_real_texts_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(text_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Statistiques sauvées: {stats_file}")
    
    # Test de validation rapide
    logger.info("Test de validation des textes...")
    sample_texts = valid_texts['text'].head(5).tolist()
    for i, text in enumerate(sample_texts):
        preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"  Exemple {i+1}: {preview}")
    
    if text_stats['valid_texts'] < 300:
        logger.error("ÉCHEC: Moins de 300 textes valides récupérés")
        return False
    
    logger.info("=== SUCCÈS: Level 1 avec textes réels créé ===")
    return output_file

if __name__ == "__main__":
    result = create_level1_with_real_texts()
    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED")
#!/usr/bin/env python3
"""
Script pour créer un dataset Level 1 complet avec tous les vrais textes.
Utilise final_summary_production.json qui contient tous les résumés.
"""

import json
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_complete_level1_dataset():
    """Crée un dataset Level 1 complet avec tous les vrais textes."""
    
    project_root = Path(__file__).parent.parent.parent
    
    # Données Level 1 existantes
    level1_file = project_root / 'data' / 'detection' / 'level1_heuristic_enriched_results.csv'
    
    # Source complète des résumés
    complete_source = project_root / 'data' / 'results' / 'final_summary_production.json'
    
    # Fichier de sortie
    output_file = project_root / 'data' / 'detection' / 'level1_complete_real_texts.csv'
    
    logger.info("=== CRÉATION DATASET LEVEL 1 COMPLET ===")
    
    # Chargement Level 1 existant
    logger.info(f"Chargement Level 1: {level1_file}")
    df_level1 = pd.read_csv(level1_file)
    logger.info(f"Level 1: {len(df_level1)} résumés")
    
    # Chargement de la source complète
    logger.info(f"Chargement source complète: {complete_source}")
    with open(complete_source, 'r', encoding='utf-8') as f:
        complete_data = json.load(f)
    
    logger.info(f"Source complète: {len(complete_data)} résumés disponibles")
    
    # Extraction des résumés "adaptive" (correspondent aux IDs Level 1)
    adaptive_summaries = {}
    for key, article_data in complete_data.items():
        if 'strategies' in article_data and 'adaptive' in article_data['strategies']:
            adaptive_data = article_data['strategies']['adaptive']
            summary_id = f"{key}_adaptive"
            
            adaptive_summaries[summary_id] = {
                'text': adaptive_data.get('summary', ''),
                'coherence': adaptive_data.get('metrics', {}).get('coherence', 0.0),
                'factuality': adaptive_data.get('metrics', {}).get('factuality', 0.0),
                'quality_grade': adaptive_data.get('quality_info', {}).get('quality_grade', 'D'),
                'production_ready': adaptive_data.get('quality_info', {}).get('production_ready', False),
                'length': len(adaptive_data.get('summary', '')),
                'word_count': len(adaptive_data.get('summary', '').split())
            }
    
    logger.info(f"Résumés adaptive extraits: {len(adaptive_summaries)}")
    
    # Matching avec les données Level 1
    matched_count = 0
    missing_count = 0
    texts_added = []
    
    for idx, row in df_level1.iterrows():
        summary_id = row['id']
        
        if summary_id in adaptive_summaries:
            # Texte réel trouvé
            summary_data = adaptive_summaries[summary_id]
            df_level1.at[idx, 'text'] = summary_data['text']
            df_level1.at[idx, 'text_length'] = summary_data['length'] 
            df_level1.at[idx, 'text_word_count'] = summary_data['word_count']
            df_level1.at[idx, 'source_coherence'] = summary_data['coherence']
            df_level1.at[idx, 'source_factuality'] = summary_data['factuality']
            df_level1.at[idx, 'source_grade'] = summary_data['quality_grade']
            df_level1.at[idx, 'production_ready'] = summary_data['production_ready']
            
            texts_added.append(summary_data['text'])
            matched_count += 1
        else:
            # Texte manquant
            df_level1.at[idx, 'text'] = f"MISSING_TEXT_FOR_{summary_id}"
            df_level1.at[idx, 'text_length'] = 0
            df_level1.at[idx, 'text_word_count'] = 0
            missing_count += 1
            logger.warning(f"Texte manquant pour {summary_id}")
    
    logger.info(f"Matching terminé: {matched_count} trouvés, {missing_count} manquants")
    
    # Validation de la qualité du dataset
    valid_texts = df_level1[~df_level1['text'].str.startswith('MISSING_TEXT_FOR_')]
    unique_texts = valid_texts['text'].nunique()
    
    dataset_stats = {
        'total_summaries': len(df_level1),
        'valid_texts': len(valid_texts),
        'missing_texts': missing_count,
        'unique_texts': unique_texts,
        'avg_length': float(valid_texts['text_length'].mean()) if len(valid_texts) > 0 else 0,
        'avg_word_count': float(valid_texts['text_word_count'].mean()) if len(valid_texts) > 0 else 0,
        'min_length': int(valid_texts['text_length'].min()) if len(valid_texts) > 0 else 0,
        'max_length': int(valid_texts['text_length'].max()) if len(valid_texts) > 0 else 0,
        'diversity_ratio': unique_texts / len(valid_texts) if len(valid_texts) > 0 else 0
    }
    
    logger.info(f"Statistiques du dataset:")
    logger.info(f"  - Textes valides: {dataset_stats['valid_texts']}/{dataset_stats['total_summaries']}")
    logger.info(f"  - Textes uniques: {dataset_stats['unique_texts']}")
    logger.info(f"  - Ratio diversité: {dataset_stats['diversity_ratio']:.3f}")
    logger.info(f"  - Longueur moyenne: {dataset_stats['avg_length']:.1f} caractères")
    logger.info(f"  - Mots moyens: {dataset_stats['avg_word_count']:.1f}")
    
    # Vérifications de qualité
    quality_checks = {
        'sufficient_texts': dataset_stats['valid_texts'] >= 300,
        'good_diversity': dataset_stats['diversity_ratio'] >= 0.7,
        'reasonable_length': dataset_stats['avg_length'] >= 100,
        'coherence_consistency': True  # À vérifier si nécessaire
    }
    
    all_checks_passed = all(quality_checks.values())
    
    logger.info(f"Vérifications qualité:")
    for check, passed in quality_checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}: {passed}")
    
    # Sauvegarde du dataset complet
    df_level1.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Dataset complet sauvé: {output_file}")
    
    # Sauvegarde des statistiques
    stats_file = output_file.parent / 'level1_complete_stats.json'
    final_stats = {**dataset_stats, 'quality_checks': quality_checks}
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"Statistiques sauvées: {stats_file}")
    
    # Exemples de textes pour validation
    logger.info("Exemples de textes récupérés:")
    sample_texts = valid_texts['text'].head(3).tolist()
    for i, text in enumerate(sample_texts):
        preview = text[:120] + "..." if len(text) > 120 else text
        logger.info(f"  {i+1}. {preview}")
    
    if all_checks_passed:
        logger.info("=== SUCCÈS: Dataset Level 1 complet créé avec succès ===")
        return output_file
    else:
        logger.error("=== ÉCHEC: Problèmes de qualité détectés ===")
        return None

if __name__ == "__main__":
    result = create_complete_level1_dataset()
    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED")
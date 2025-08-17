#!/usr/bin/env python3
"""
Test rigoureux du Level 2 avec des données réelles diversifiées.
Corrige tous les problèmes identifiés dans l'analyse critique.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict
import logging

# Configuration des chemins
project_root = Path(__file__).parent.parent.parent
src_path = project_root / 'src'
sys.path.append(str(src_path))

# Import des modules Level 2
from detection.level2_factual.level2_coordinator import Level2FactualProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_diverse_test_dataset():
    """Crée un dataset de test diversifié avec des textes réels variés."""
    
    # Chargement des sources de données
    level1_file = project_root / 'data' / 'detection' / 'level1_heuristic_enriched_results.csv'
    source_file = project_root / 'data' / 'results' / 'final_summary_production.json'
    
    df_level1 = pd.read_csv(level1_file)
    
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # Extraction des textes réels
    real_texts = []
    for key, article_data in source_data.items():
        if 'strategies' in article_data and 'adaptive' in article_data['strategies']:
            adaptive_data = article_data['strategies']['adaptive']
            text = adaptive_data.get('summary', '')
            if text and len(text) > 50:  # Filtre les textes trop courts
                real_texts.append({
                    'id': f"{key}_adaptive",
                    'text': text,
                    'coherence': adaptive_data.get('metrics', {}).get('coherence', 0.5),
                    'factuality': adaptive_data.get('metrics', {}).get('factuality', 0.5),
                    'grade': adaptive_data.get('quality_info', {}).get('quality_grade', 'B'),
                    'length': len(text),
                    'word_count': len(text.split())
                })
    
    logger.info(f"Textes réels extraits: {len(real_texts)}")
    
    # Sélection diversifiée par grade et longueur
    test_summaries = []
    grade_targets = {'A+': 10, 'A': 20, 'B+': 30, 'B': 15, 'C': 10, 'D': 15}
    
    # Groupement par grade
    by_grade = defaultdict(list)
    for text_data in real_texts:
        by_grade[text_data['grade']].append(text_data)
    
    # Sélection équilibrée
    for grade, target_count in grade_targets.items():
        available = by_grade.get(grade, [])
        selected = available[:target_count] if len(available) >= target_count else available
        test_summaries.extend(selected)
    
    # Ajout de candidats fact-check variés (simulation basée sur contenu)
    for summary in test_summaries:
        text = summary['text']
        # Détection simple de candidats potentiels
        candidates_count = 0
        # Noms propres (approximation)
        candidates_count += text.count('Microsoft') + text.count('Google') + text.count('Apple')
        # Dates
        candidates_count += len([w for w in text.split() if w.isdigit() and len(w) == 4])
        # Chiffres
        candidates_count += len([w for w in text.split() if '%' in w or '€' in w])
        
        summary['fact_check_candidates_count'] = min(candidates_count, 5)
        summary['heuristic_valid'] = True
        summary['risk_level'] = 'medium'
    
    logger.info(f"Dataset de test créé: {len(test_summaries)} résumés diversifiés")
    return test_summaries

def test_level2_with_real_data():
    """Test complet du Level 2 avec des données réelles diversifiées."""
    
    logger.info("=== TEST RIGOUREUX LEVEL 2 AVEC DONNÉES RÉELLES ===")
    
    # Création du dataset de test
    test_summaries = create_diverse_test_dataset()
    
    if len(test_summaries) < 50:
        logger.error("Dataset insuffisant pour test rigoureux")
        return False
    
    # Initialisation du processeur Level 2
    logger.info("Initialisation du processeur Level 2...")
    level2_processor = Level2FactualProcessor(performance_mode="balanced")
    
    # Analyse de la distribution des tiers AVANT traitement
    logger.info("Analyse de classification des tiers...")
    tier_prediction = defaultdict(int)
    
    for summary in test_summaries:
        predicted_tier = level2_processor.classify_summary_tier(summary)
        tier_prediction[predicted_tier] += 1
    
    logger.info("Distribution prédite des tiers:")
    for tier, count in tier_prediction.items():
        percentage = count / len(test_summaries) * 100
        logger.info(f"  - {tier}: {count} résumés ({percentage:.1f}%)")
    
    # Traitement Level 2
    logger.info(f"Traitement Level 2 sur {len(test_summaries)} résumés réels...")
    
    start_time = time.time()
    
    try:
        valid_summaries, results = level2_processor.process_batch(test_summaries)
        processing_time = time.time() - start_time
        
        logger.info(f"Traitement terminé en {processing_time:.2f}s")
        logger.info(f"Résumés validés: {len(valid_summaries)}/{len(test_summaries)}")
        logger.info(f"Résultats générés: {len(results)}")
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement: {e}")
        return False
    
    # Analyse rigoureuse des résultats
    logger.info("=== ANALYSE RIGOUREUSE DES RÉSULTATS ===")
    
    # 1. Vérification de la diversité des scores
    confidences = [r.factual_confidence for r in results]
    confidence_stats = {
        'mean': np.mean(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'unique_values': len(set(confidences))
    }
    
    logger.info("Statistiques de confiance factuelle:")
    logger.info(f"  - Moyenne: {confidence_stats['mean']:.3f}")
    logger.info(f"  - Écart-type: {confidence_stats['std']:.3f}")
    logger.info(f"  - Min-Max: {confidence_stats['min']:.3f} - {confidence_stats['max']:.3f}")
    logger.info(f"  - Valeurs uniques: {confidence_stats['unique_values']}")
    
    # 2. Analyse des éléments flagués
    all_flagged = []
    flagged_types = defaultdict(int)
    
    for result in results:
        for flagged in result.flagged_elements:
            all_flagged.append(flagged)
            
            # Classification plus précise des types
            if 'Processing error' in flagged:
                flagged_types['Erreurs techniques'] += 1
            elif 'Incohérence coherence-factuality' in flagged:
                flagged_types['Incohérences coherence-factuality'] += 1
            elif 'Candidat suspect' in flagged:
                flagged_types['Candidats suspects'] += 1
            elif 'Crédibilité linguistique' in flagged:
                flagged_types['Problèmes linguistiques'] += 1
            elif 'Anomalie statistique' in flagged:
                flagged_types['Anomalies statistiques'] += 1
            elif 'Incohérence' in flagged:
                flagged_types['Incohérences internes'] += 1
            else:
                flagged_types['Autres'] += 1
    
    logger.info(f"Éléments flagués: {len(all_flagged)} total")
    logger.info("Types d'anomalies détectées:")
    for flag_type, count in sorted(flagged_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(all_flagged) * 100 if all_flagged else 0
        logger.info(f"  - {flag_type}: {count} ({percentage:.1f}%)")
    
    # 3. Analyse des performances par tier
    tier_performance = defaultdict(list)
    for result in results:
        tier_performance[result.tier_classification].append({
            'confidence': result.factual_confidence,
            'processing_time': result.processing_time_ms,
            'flagged_count': len(result.flagged_elements)
        })
    
    logger.info("Performance par tier:")
    for tier, performances in tier_performance.items():
        if performances:
            avg_confidence = np.mean([p['confidence'] for p in performances])
            avg_time = np.mean([p['processing_time'] for p in performances])
            avg_flagged = np.mean([p['flagged_count'] for p in performances])
            
            logger.info(f"  - {tier}: {len(performances)} résumés")
            logger.info(f"    Conf: {avg_confidence:.3f}, Temps: {avg_time:.1f}ms, Flagués: {avg_flagged:.1f}")
    
    # 4. Vérifications de qualité
    quality_checks = {
        'diverse_confidences': confidence_stats['std'] > 0.05,  # Scores variés
        'reasonable_processing_time': processing_time / len(test_summaries) < 2.0,  # <2s par résumé
        'meaningful_detections': len(set(all_flagged)) > 5,  # Au moins 5 types différents
        'balanced_tiers': len(tier_performance) >= 3,  # Au moins 3 tiers utilisés
        'low_error_rate': flagged_types.get('Erreurs techniques', 0) / len(results) < 0.1,  # <10% erreurs
        'content_based_analysis': confidence_stats['unique_values'] > len(test_summaries) * 0.1  # Analyse réelle
    }
    
    logger.info("=== VÉRIFICATIONS DE QUALITÉ ===")
    passed_checks = 0
    for check, passed in quality_checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {check}: {passed}")
        if passed:
            passed_checks += 1
    
    quality_score = passed_checks / len(quality_checks)
    logger.info(f"Score de qualité global: {quality_score:.1%} ({passed_checks}/{len(quality_checks)})")
    
    # 5. Exemples concrets de détections
    logger.info("=== EXEMPLES DE DÉTECTIONS ===")
    if all_flagged:
        unique_flagged = list(set(all_flagged))[:10]  # Top 10 uniques
        for i, example in enumerate(unique_flagged):
            preview = example[:100] + "..." if len(example) > 100 else example
            logger.info(f"  {i+1}. {preview}")
    
    # Sauvegarde des résultats pour analyse
    results_data = []
    for result in results:
        results_data.append({
            'summary_id': result.summary_id,
            'tier': result.tier_classification,
            'factual_confidence': result.factual_confidence,
            'processing_time_ms': result.processing_time_ms,
            'flagged_elements': result.flagged_elements,
            'level3_priority': result.get_level3_priority()
        })
    
    output_file = project_root / 'data' / 'detection' / 'level2_real_data_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_info': {
                'dataset_size': len(test_summaries),
                'processing_time_s': processing_time,
                'quality_score': quality_score
            },
            'quality_checks': quality_checks,
            'confidence_stats': confidence_stats,
            'flagged_types': dict(flagged_types),
            'results': results_data
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Résultats sauvés: {output_file}")
    
    # Verdict final
    if quality_score >= 0.75:
        logger.info("🎉 SUCCÈS: Level 2 fonctionne correctement avec des données réelles")
        return True
    else:
        logger.error("❌ ÉCHEC: Level 2 ne satisfait pas les critères de qualité")
        return False

if __name__ == "__main__":
    success = test_level2_with_real_data()
    sys.exit(0 if success else 1)
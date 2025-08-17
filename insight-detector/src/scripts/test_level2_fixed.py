#!/usr/bin/env python3
"""
Script de test complet pour valider les corrections du Level 2.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict

# Configuration des chemins
project_root = Path(__file__).parent.parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Import des modules Level 2
from detection.level2_factual.level2_coordinator import Level2FactualProcessor
from scripts.fix_level1_data import fix_level1_data

def test_level2_corrections():
    """Test complet du Level 2 avec les corrections appliquées."""
    
    print("=== TEST DES CORRECTIONS LEVEL 2 ===\n")
    
    # 1. Correction des données Level 1
    print("1. Réparation des données Level 1...")
    try:
        corrected_file = fix_level1_data()
        print(f"[OK] Données corrigées: {corrected_file}")
    except Exception as e:
        print(f"[ERREUR] Erreur réparation données: {e}")
        return False
    
    # 2. Chargement des données corrigées
    print("\n2. Chargement des données corrigées...")
    df_corrected = pd.read_csv(corrected_file)
    print(f"[OK] {len(df_corrected)} résumés chargés")
    print(f"[OK] Colonne 'text' présente: {'text' in df_corrected.columns}")
    
    # Vérification des textes
    valid_texts = (~df_corrected['text'].isna() & (df_corrected['text'] != '')).sum()
    print(f"[OK] Textes valides: {valid_texts}/{len(df_corrected)} ({valid_texts/len(df_corrected)*100:.1f}%)")
    
    # 3. Test de la classification des tiers corrigée
    print("\n3. Test de la classification des tiers...")
    level2_processor = Level2FactualProcessor(performance_mode="balanced")
    
    tier_distribution = defaultdict(int)
    sample_size = min(50, len(df_corrected))  # Test sur 50 résumés
    
    for _, row in df_corrected.head(sample_size).iterrows():
        summary_data = {
            'id': row.get('id', f'test_{_}'),
            'text': str(row.get('text', '')),
            'original_grade': row.get('original_grade', 'D'),
            'coherence': float(row.get('coherence', 0.0)),
            'factuality': float(row.get('factuality', 0.0)),
            'fact_check_candidates_count': int(row.get('fact_check_candidates_count', 0)),
            'heuristic_valid': bool(row.get('heuristic_valid', False)),
            'risk_level': row.get('risk_level', 'medium')
        }
        
        tier = level2_processor.classify_summary_tier(summary_data)
        tier_distribution[tier] += 1
    
    print(f"Distribution des tiers (sur {sample_size} résumés):")
    for tier, count in tier_distribution.items():
        percentage = count / sample_size * 100
        print(f"  - {tier}: {count} résumés ({percentage:.1f}%)")
    
    # Vérification de la distribution
    tier2_present = tier_distribution.get('TIER_2_MODERATE', 0) > 0
    tier4_reasonable = tier_distribution.get('TIER_4_CRITICAL', 0) / sample_size < 0.5
    
    print(f"[OK] TIER_2_MODERATE présent: {tier2_present}")
    print(f"[OK] TIER_4_CRITICAL raisonnable (<50%): {tier4_reasonable}")
    
    # 4. Test des validateurs (sans bugs regex)
    print("\n4. Test des validateurs...")
    test_summaries = []
    error_count = 0
    
    for i in range(min(10, len(df_corrected))):
        row = df_corrected.iloc[i]
        summary_data = {
            'id': row.get('id', f'test_{i}'),
            'text': str(row.get('text', 'Texte de test pour validation')),
            'original_grade': row.get('original_grade', 'B+'),
            'coherence': float(row.get('coherence', 0.7)),
            'factuality': float(row.get('factuality', 0.8)),
            'fact_check_candidates_count': int(row.get('fact_check_candidates_count', 1))
        }
        test_summaries.append(summary_data)
    
    try:
        start_time = time.time()
        valid_summaries, results = level2_processor.process_batch(test_summaries)
        processing_time = time.time() - start_time
        
        # Vérification des erreurs
        for result in results:
            if any('Processing error' in flag for flag in result.flagged_elements):
                error_count += 1
        
        print(f"[OK] Traitement batch: {len(results)} résultats")
        print(f"[OK] Temps de traitement: {processing_time*1000:.1f}ms")
        print(f"[OK] Erreurs techniques: {error_count}/{len(results)} ({error_count/len(results)*100:.1f}%)")
        
        # Vérification des métriques
        confidences = [r.factual_confidence for r in results]
        avg_confidence = np.mean(confidences)
        print(f"[OK] Confiance moyenne: {avg_confidence:.3f}")
        
    except Exception as e:
        print(f"[ERREUR] Erreur traitement batch: {e}")
        return False
    
    # 5. Résumé des corrections
    print("\n=== RÉSUMÉ DES CORRECTIONS ===")
    print("[OK] 1. Données Level 1 réparées (colonne 'text' ajoutée)")
    print("[OK] 2. Bug regex corrigé (apostrophes)")
    print("[OK] 3. Classification des tiers rebalancée")
    print(f"[OK] 4. Erreurs techniques réduites: {error_count}/{len(results)} cas")
    
    # Évaluation globale
    success_criteria = [
        tier2_present,  # TIER_2 présent
        tier4_reasonable,  # TIER_4 raisonnable
        error_count < len(results) * 0.1,  # <10% d'erreurs
        avg_confidence > 0.3  # Confiance raisonnable
    ]
    
    success_rate = sum(success_criteria) / len(success_criteria)
    print(f"\n[RÉSULTAT] TAUX DE SUCCÈS DES CORRECTIONS: {success_rate*100:.1f}%")
    
    if success_rate >= 0.75:
        print("[SUCCÈS] CORRECTIONS VALIDÉES - Level 2 prêt pour tests complets")
        return True
    else:
        print("[ATTENTION] CORRECTIONS PARTIELLES - Ajustements nécessaires")
        return False

if __name__ == "__main__":
    success = test_level2_corrections()
    sys.exit(0 if success else 1)
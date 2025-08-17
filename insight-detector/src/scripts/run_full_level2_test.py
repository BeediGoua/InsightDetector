#!/usr/bin/env python3
"""
Test complet du Level 2 sur les 372 résumés avec les corrections appliquées.
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

def run_full_level2_test():
    """Test complet du Level 2 sur tous les 372 résumés avec corrections."""
    
    print("=== TEST COMPLET LEVEL 2 CORRIGÉ ===\n")
    
    # Chargement des données corrigées
    data_file = project_root / 'data' / 'detection' / 'level1_with_texts.csv'
    
    if not data_file.exists():
        print("[ERREUR] Données corrigées non trouvées. Exécutez d'abord test_level2_fixed.py")
        return False
    
    print("1. Chargement des données corrigées...")
    df = pd.read_csv(data_file)
    print(f"[OK] {len(df)} résumés chargés avec textes réels")
    
    # Préparation des données
    print("\n2. Préparation des données...")
    all_summaries = []
    
    for _, row in df.iterrows():
        summary_data = {
            'id': row.get('id', f'summary_{_}'),
            'text': str(row.get('text', '')),
            'original_grade': row.get('original_grade', 'D'),
            'coherence': float(row.get('coherence', 0.0)),
            'factuality': float(row.get('factuality', 0.0)),
            'fact_check_candidates_count': int(row.get('fact_check_candidates_count', 0)),
            'heuristic_valid': bool(row.get('heuristic_valid', False)),
            'risk_level': row.get('risk_level', 'medium')
        }
        all_summaries.append(summary_data)
    
    print(f"[OK] {len(all_summaries)} résumés préparés")
    
    # Initialisation du processeur Level 2
    print("\n3. Initialisation Level 2...")
    level2_processor = Level2FactualProcessor(performance_mode="balanced")
    
    # Classification préliminaire des tiers
    tier_distribution = defaultdict(int)
    for summary in all_summaries:
        tier = level2_processor.classify_summary_tier(summary)
        tier_distribution[tier] += 1
    
    print("Distribution des tiers prévue:")
    for tier, count in tier_distribution.items():
        percentage = count / len(all_summaries) * 100
        print(f"  - {tier}: {count} résumés ({percentage:.1f}%)")
    
    # Traitement par batch
    print(f"\n4. Traitement complet (par batch de 50)...")
    batch_size = 50
    all_results = []
    total_batches = (len(all_summaries) + batch_size - 1) // batch_size
    
    start_time = time.time()
    error_count = 0
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(all_summaries))
        batch_summaries = all_summaries[batch_start:batch_end]
        
        try:
            valid_summaries, results = level2_processor.process_batch(batch_summaries)
            all_results.extend(results)
            
            # Comptage des erreurs
            batch_errors = sum(1 for r in results if any('Processing error' in flag for flag in r.flagged_elements))
            error_count += batch_errors
            
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%): {len(results)} résultats, {batch_errors} erreurs")
            
        except Exception as e:
            print(f"  [ERREUR] Batch {batch_idx + 1}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Analyse des résultats
    print(f"\n5. Analyse des résultats...")
    print(f"[OK] Temps total: {total_time:.2f}s ({total_time/len(all_summaries)*1000:.1f}ms/résumé)")
    print(f"[OK] Résultats générés: {len(all_results)}/{len(all_summaries)}")
    print(f"[OK] Erreurs techniques: {error_count}/{len(all_results)} ({error_count/len(all_results)*100:.1f}%)")
    
    # Métriques de qualité
    confidences = [r.factual_confidence for r in all_results]
    flagged_counts = [len(r.flagged_elements) for r in all_results]
    
    print(f"\nMétriques de qualité:")
    print(f"  - Confiance moyenne: {np.mean(confidences):.3f}")
    print(f"  - Confiance médiane: {np.median(confidences):.3f}")
    print(f"  - Écart-type confiance: {np.std(confidences):.3f}")
    print(f"  - Éléments flagués moyens: {np.mean(flagged_counts):.1f}")
    print(f"  - Total éléments flagués: {sum(flagged_counts)}")
    
    # Distribution des risques
    risk_distribution = defaultdict(int)
    for result in all_results:
        risk_distribution[result.factual_risk_level] += 1
    
    print(f"\nDistribution des risques:")
    for risk, count in risk_distribution.items():
        percentage = count / len(all_results) * 100
        print(f"  - {risk}: {count} résumés ({percentage:.1f}%)")
    
    # Priorisation Level 3
    high_priority = sum(1 for r in all_results if r.get_level3_priority() > 0.7)
    medium_priority = sum(1 for r in all_results if 0.4 <= r.get_level3_priority() <= 0.7)
    low_priority = sum(1 for r in all_results if r.get_level3_priority() < 0.4)
    
    print(f"\nPriorisation Level 3:")
    print(f"  - Priorité élevée (>0.7): {high_priority} résumés ({high_priority/len(all_results)*100:.1f}%)")
    print(f"  - Priorité moyenne (0.4-0.7): {medium_priority} résumés ({medium_priority/len(all_results)*100:.1f}%)")
    print(f"  - Priorité faible (<0.4): {low_priority} résumés ({low_priority/len(all_results)*100:.1f}%)")
    
    # Sauvegarde des résultats
    print(f"\n6. Sauvegarde des résultats...")
    output_path = project_root / 'data' / 'detection'
    
    # DataFrame des résultats
    results_data = []
    for result in all_results:
        results_data.append({
            'summary_id': result.summary_id,
            'tier': result.tier_classification,
            'factual_confidence': result.factual_confidence,
            'risk_level': result.factual_risk_level,
            'processing_time_ms': result.processing_time_ms,
            'num_flagged_elements': len(result.flagged_elements),
            'level3_priority': result.get_level3_priority()
        })
    
    df_results = pd.DataFrame(results_data)
    results_file = output_path / 'level2_corrected_results.csv'
    df_results.to_csv(results_file, index=False, encoding='utf-8')
    print(f"[OK] Résultats sauvés: {results_file}")
    
    # Statistiques finales
    final_stats = {
        'test_info': {
            'dataset_size': len(all_summaries),
            'total_processing_time_s': total_time,
            'avg_time_per_summary_ms': total_time / len(all_summaries) * 1000
        },
        'corrections_applied': {
            'data_fixed': True,
            'regex_bug_fixed': True,
            'tier_classification_rebalanced': True
        },
        'performance_metrics': {
            'error_rate': error_count / len(all_results),
            'avg_confidence': float(np.mean(confidences)),
            'median_confidence': float(np.median(confidences))
        },
        'tier_distribution': dict(tier_distribution),
        'risk_distribution': dict(risk_distribution),
        'level3_prioritization': {
            'high_priority': int(high_priority),
            'medium_priority': int(medium_priority),
            'low_priority': int(low_priority)
        }
    }
    
    stats_file = output_path / 'level2_corrected_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)
    print(f"[OK] Statistiques sauvées: {stats_file}")
    
    # Évaluation finale
    print(f"\n=== ÉVALUATION FINALE ===")
    objectives_met = {
        'Latence <1s': total_time / len(all_summaries) < 1.0,
        'Erreurs <10%': error_count / len(all_results) < 0.1,
        'TIER_2 présent': tier_distribution.get('TIER_2_MODERATE', 0) > 0,
        'TIER_4 raisonnable': tier_distribution.get('TIER_4_CRITICAL', 0) / len(all_summaries) < 0.3,
        'Confiance >0.5': np.mean(confidences) > 0.5
    }
    
    met_count = sum(objectives_met.values())
    success_rate = met_count / len(objectives_met)
    
    print(f"Objectifs Level 2 atteints:")
    for obj, met in objectives_met.items():
        status = "[OK]" if met else "[KO]"
        print(f"  {status} {obj}")
    
    print(f"\n[RÉSULTAT] TAUX DE RÉUSSITE: {success_rate*100:.1f}% ({met_count}/{len(objectives_met)})")
    
    if success_rate >= 0.8:
        print("[SUCCÈS] Level 2 opérationnel et prêt pour production")
        return True
    else:
        print("[ATTENTION] Ajustements supplémentaires recommandés")
        return False

if __name__ == "__main__":
    success = run_full_level2_test()
    sys.exit(0 if success else 1)
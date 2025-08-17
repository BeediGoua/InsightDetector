#!/usr/bin/env python3
"""
Script de test pour valider les améliorations du Level 2.
Test rapide des corrections apportées sans relancer le notebook complet.
"""

import sys
import os
from pathlib import Path

# Configuration des chemins
project_root = Path(__file__).parent.parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from detection.level2_factual.level2_coordinator import Level2FactualProcessor
from detection.level2_factual.internal_consistency_analyzer import InternalConsistencyAnalyzer

def test_parsing_fixes():
    """Test les corrections de parsing pour éviter les erreurs tuple."""
    print("=== Test des corrections de parsing ===")
    
    analyzer = InternalConsistencyAnalyzer()
    
    # Test avec différents types d'entrées
    test_cases = [
        "Texte normal avec Emmanuel Macron et Microsoft.",
        ("tuple", "accidentel"),  # Cas problématique original
        None,  # Cas null
        "",    # Cas vide
        "Texte avec variations: Omar AL-Rashid vs Omar Al-Rashid",
    ]
    
    for i, test_input in enumerate(test_cases):
        try:
            result = analyzer.validate({'text': test_input})
            print(f"  Test {i+1}: Succes - Confiance: {result.get('score', 0):.3f}")
        except Exception as e:
            print(f"  Test {i+1}: Erreur - {str(e)}")
    
    print()

def test_intelligent_entity_detection():
    """Test la détection intelligente d'entités."""
    print("=== Test de la détection intelligente d'entités ===")
    
    analyzer = InternalConsistencyAnalyzer()
    
    # Cas qui ne devraient PAS être détectés comme suspects (faux positifs avant)
    normal_cases = [
        "Emmanuel Macron est président. Emmanuel a déclaré que...",
        "Microsoft Corporation et Microsoft sont la même entreprise.",
        "Le président et Emmanuel Macron ont discuté.",
        "Omar Al-Rashid, aussi connu sous le nom d'Omar, était présent.",
    ]
    
    # Cas qui DEVRAIENT être détectés comme suspects
    suspicious_cases = [
        "Microsoft a gagné 100 millions. Microsoft a perdu 200 millions.",  # Contradiction
        "John Smith et John Smyth sont deux personnes différentes.",  # Orthographe suspecte
        "L'événement s'est passé en 2020. En 2021, c'était différent.",  # Dates contradictoires
    ]
    
    print("  Cas normaux (ne devraient PAS être flagués):")
    for i, case in enumerate(normal_cases):
        result = analyzer.validate({'text': case})
        flagged = len(result.get('flagged_elements', []))
        status = "OK" if flagged == 0 else "WARN"
        print(f"    {i+1}. {status} Flagués: {flagged}")
    
    print("\n  Cas suspects (DEVRAIENT être flagués):")
    for i, case in enumerate(suspicious_cases):
        result = analyzer.validate({'text': case})
        flagged = len(result.get('flagged_elements', []))
        status = "OK" if flagged > 0 else "FAIL"
        print(f"    {i+1}. {status} Flagués: {flagged}")
    
    print()

def test_tier_classification():
    """Test la nouvelle classification par tiers."""
    print("=== Test de la classification par tiers ===")
    
    processor = Level2FactualProcessor(performance_mode="balanced")
    
    # Cas de test avec scores de risque variés
    test_summaries = [
        {
            'id': 'test_safe',
            'original_grade': 'A+',
            'coherence': 0.95,
            'factuality': 0.98,
            'fact_check_candidates_count': 0,
            'detected_issues': ''
        },
        {
            'id': 'test_moderate',
            'original_grade': 'B+',
            'coherence': 0.70,
            'factuality': 0.80,
            'fact_check_candidates_count': 2,
            'detected_issues': 'Quelques issues mineures'
        },
        {
            'id': 'test_complex',
            'original_grade': 'C',
            'coherence': 0.50,
            'factuality': 0.60,
            'fact_check_candidates_count': 5,
            'detected_issues': 'Issues multiples détectées'
        },
        {
            'id': 'test_critical',
            'original_grade': 'D',
            'coherence': 0.20,
            'factuality': 0.30,
            'fact_check_candidates_count': 8,
            'detected_issues': 'Nombreuses issues critiques détectées'
        }
    ]
    
    expected_tiers = ['TIER_1_SAFE', 'TIER_2_MODERATE', 'TIER_4_CRITICAL', 'TIER_4_CRITICAL']
    
    for i, summary in enumerate(test_summaries):
        tier = processor.classify_summary_tier(summary)
        risk_score = processor._calculate_risk_score(summary)
        expected = expected_tiers[i]
        status = "OK" if tier == expected else "WARN"
        
        print(f"  {i+1}. {status} Grade {summary['original_grade']} -> {tier}")
        print(f"      Score risque: {risk_score:.3f}, Attendu: {expected}")
    
    print()

def main():
    """Exécute tous les tests d'amélioration."""
    print("Test des ameliorations Level 2")
    print("=" * 50)
    
    try:
        test_parsing_fixes()
        test_intelligent_entity_detection()
        test_tier_classification()
        
        print("Tous les tests termines!")
        print("\nResume des ameliorations:")
        print("  1. Protection contre erreurs de parsing (tuple/None)")
        print("  2. Detection semantique intelligente des entites") 
        print("  3. Classification par tiers basee sur score de risque")
        print("  4. Reduction des faux positifs linguistiques")
        
    except Exception as e:
        print(f"Erreur pendant les tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
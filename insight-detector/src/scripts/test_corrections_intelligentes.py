#!/usr/bin/env python3
"""
Test rigoureux des corrections intelligentes apportées au Level 2.
Validation ciblée des problèmes identifiés.
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

def test_correction_variations_francaises():
    """Test spécifique des corrections de variations françaises."""
    print("=== Test Correction Variations Françaises ===")
    
    analyzer = InternalConsistencyAnalyzer()
    
    # Cas qui NE DEVRAIENT PLUS être détectés comme suspects
    normal_cases = [
        ("imprudente", "imprudente ne"),  # négation normale
        ("virées", "virées de"),          # préposition normale  
        ("possible", "possible de"),      # construction infinitive
        ("dans", "dans le"),              # article défini
        ("Microsoft", "Microsoft Corporation"),  # expansion normale
    ]
    
    print("  Variations normales (ne devraient PAS être suspectes):")
    false_positives = 0
    for var1, var2 in normal_cases:
        is_suspicious = analyzer._is_suspicious_entity_variation(var1, var2)
        status = "FAIL" if is_suspicious else "OK"
        if is_suspicious:
            false_positives += 1
        print(f"    '{var1}' vs '{var2}': {status}")
    
    # Cas qui DEVRAIENT être détectés comme suspects
    suspicious_cases = [
        ("Microsoft", "Microsooft"),      # faute d'orthographe
        ("2020", "2021"),                 # différence de chiffres
        ("Jean Dupont", "Jean Durand"),   # noms différents
        ("100 millions", "200 millions"), # chiffres contradictoires
    ]
    
    print("\n  Variations suspectes (DEVRAIENT être détectées):")
    false_negatives = 0
    for var1, var2 in suspicious_cases:
        is_suspicious = analyzer._is_suspicious_entity_variation(var1, var2)
        status = "OK" if is_suspicious else "FAIL"
        if not is_suspicious:
            false_negatives += 1
        print(f"    '{var1}' vs '{var2}': {status}")
    
    print(f"\n  Résultats: {false_positives} faux positifs, {false_negatives} faux négatifs")
    return false_positives, false_negatives

def test_correction_contradictions():
    """Test des corrections de détection de contradictions."""
    print("\n=== Test Correction Contradictions ===")
    
    analyzer = InternalConsistencyAnalyzer()
    
    # Cas qui NE DEVRAIENT PAS être détectés comme contradictions
    normal_texts = [
        "Imaginez : vous êtes en vacances. En savoir plus sur l'utilisation des données personnelles.",
        "Partager Facebook Twitter E-mail. Mise à jour du 25 juillet.",
        "Le président a déclaré. Le président français a confirmé.",  # même entité
    ]
    
    print("  Textes normaux (pas de contradictions attendues):")
    false_contradictions = 0
    for text in normal_texts:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) >= 2:
            contradiction = analyzer._check_sentence_contradiction(sentences[0], sentences[1])
            status = "FAIL" if contradiction else "OK"
            if contradiction:
                false_contradictions += 1
            print(f"    {status}: '{text[:50]}...'")
    
    # Cas qui DEVRAIENT être détectés comme contradictions
    contradictory_texts = [
        ("Microsoft a gagné 100 millions d'euros.", "Microsoft a perdu 200 millions d'euros."),
        ("Le projet est terminé avec succès.", "Le projet n'a jamais abouti et reste inachevé."),
        ("L'entreprise emploie 5000 personnes.", "L'entreprise n'emploie que 2000 salariés."),
    ]
    
    print("\n  Vraies contradictions (DEVRAIENT être détectées):")
    missed_contradictions = 0
    for sent1, sent2 in contradictory_texts:
        contradiction = analyzer._check_sentence_contradiction(sent1, sent2)
        status = "OK" if contradiction else "FAIL"
        if not contradiction:
            missed_contradictions += 1
        print(f"    {status}: '{sent1}' vs '{sent2}'")
    
    print(f"\n  Résultats: {false_contradictions} fausses contradictions, {missed_contradictions} contradictions ratées")
    return false_contradictions, missed_contradictions

def test_correction_sensibilite_tiers():
    """Test de la sensibilité par tiers."""
    print("\n=== Test Sensibilité par Tiers ===")
    
    processor = Level2FactualProcessor(performance_mode="balanced")
    
    # Test des seuils de sensibilité
    tiers = ['TIER_1_SAFE', 'TIER_2_MODERATE', 'TIER_3_COMPLEX', 'TIER_4_CRITICAL']
    
    print("  Seuils de sensibilité:")
    for tier in tiers:
        sensitivity = processor._get_detection_sensitivity(tier)
        print(f"    {tier}: {sensitivity}")
    
    # Test du filtrage par sensibilité
    test_flagged = [
        "Incohérence entités: Variations d'entité suspectes: imprudente vs imprudente ne",
        "Contradiction: Contradiction potentielle entre: 'Microsoft a gagné' et 'Microsoft a perdu'",
        "Processing error: expected string or bytes-like object, got 'tuple'"
    ]
    
    print("\n  Filtrage par tier:")
    for tier in tiers:
        sensitivity = processor._get_detection_sensitivity(tier)
        filtered = processor._filter_flagged_by_sensitivity(test_flagged, sensitivity, tier)
        print(f"    {tier}: {len(filtered)}/{len(test_flagged)} éléments gardés")
        for element in filtered:
            print(f"      - {element[:60]}...")
    
    return True

def test_correction_classification_tiers():
    """Test de la classification par tiers corrigée."""
    print("\n=== Test Classification Tiers ===")
    
    processor = Level2FactualProcessor(performance_mode="balanced")
    
    # Cas de test représentatifs
    test_cases = [
        {
            'id': 'safe_case',
            'original_grade': 'A+',
            'coherence': 0.95,
            'factuality': 0.98,
            'fact_check_candidates_count': 0,
            'expected_tier': 'TIER_1_SAFE'
        },
        {
            'id': 'moderate_case',
            'original_grade': 'A',
            'coherence': 0.75,
            'factuality': 0.80,
            'fact_check_candidates_count': 2,
            'expected_tier': 'TIER_2_MODERATE'
        },
        {
            'id': 'complex_case',
            'original_grade': 'B',
            'coherence': 0.50,
            'factuality': 0.60,
            'fact_check_candidates_count': 4,
            'expected_tier': 'TIER_3_COMPLEX'
        },
        {
            'id': 'critical_case',
            'original_grade': 'D',
            'coherence': 0.20,
            'factuality': 0.30,
            'fact_check_candidates_count': 8,
            'expected_tier': 'TIER_4_CRITICAL'
        }
    ]
    
    print("  Classification des cas de test:")
    correct_classifications = 0
    for case in test_cases:
        actual_tier = processor.classify_summary_tier(case)
        expected_tier = case['expected_tier']
        status = "OK" if actual_tier == expected_tier else "FAIL"
        if actual_tier == expected_tier:
            correct_classifications += 1
        
        print(f"    {case['id']}: {status}")
        print(f"      Grade: {case['original_grade']}, Coherence: {case['coherence']}")
        print(f"      Attendu: {expected_tier}, Obtenu: {actual_tier}")
    
    accuracy = correct_classifications / len(test_cases) * 100
    print(f"\n  Précision classification: {correct_classifications}/{len(test_cases)} ({accuracy:.1f}%)")
    return accuracy

def main():
    """Execute tous les tests de correction."""
    print("Test Rigoureux des Corrections Intelligentes Level 2")
    print("=" * 60)
    
    try:
        # Test des corrections
        fp_entities, fn_entities = test_correction_variations_francaises()
        fp_contrad, fn_contrad = test_correction_contradictions()
        test_correction_sensibilite_tiers()
        accuracy = test_correction_classification_tiers()
        
        print("\n" + "=" * 60)
        print("BILAN GLOBAL DES CORRECTIONS:")
        print(f"  Variations françaises: {fp_entities} faux positifs, {fn_entities} faux négatifs")
        print(f"  Contradictions: {fp_contrad} fausses détections, {fn_contrad} ratées")
        print(f"  Classification: {accuracy:.1f}% de précision")
        
        # Évaluation globale
        total_errors = fp_entities + fn_entities + fp_contrad + fn_contrad
        if total_errors == 0 and accuracy >= 75:
            print("\nSTATUT: CORRECTIONS RÉUSSIES ✓")
        elif total_errors <= 2 and accuracy >= 50:
            print("\nSTATUT: CORRECTIONS PARTIELLES ⚠")
        else:
            print("\nSTATUT: CORRECTIONS INSUFFISANTES ✗")
        
        print(f"\nLes corrections sont prêtes pour test sur le notebook complet.")
        
    except Exception as e:
        print(f"Erreur pendant les tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
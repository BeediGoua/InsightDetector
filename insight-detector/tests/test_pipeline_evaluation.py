#!/usr/bin/env python3
"""
Test du système d'évaluation complet du pipeline InsightDetector.

Démontre:
- Évaluation performance par niveau (0-3)
- Métriques de qualité bout-en-bout
- Détection problèmes de robustesse
- Génération recommandations d'amélioration
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Configuration des chemins pour exécution depuis tests/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configuration logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import évaluateur
sys.path.append(str(Path(__file__).parent / 'src'))
from evaluation.pipeline_metrics import create_pipeline_evaluator


# Structures de données simulées pour test
@dataclass
class MockFilterResult:
    """Résultat simulé niveau 0."""
    is_valid: bool
    can_be_used: bool
    processing_time_ms: float
    corrections_applied: List[str]
    rejection_reasons: List[str]
    corruption_score: float
    original_word_count: int
    word_count: int


@dataclass
class MockHeuristicResult:
    """Résultat simulé niveau 1."""
    is_suspect: bool
    confidence_score: float
    risk_level: str
    issues: List[Dict]
    processing_time_ms: float
    word_count: int
    entities_detected: int
    fact_check_candidates: List[Dict]


@dataclass
class MockLevel2Result:
    """Résultat simulé niveau 2."""
    is_valid: bool
    is_production_ready: bool
    validation_confidence: float
    tier_classification: Any
    level3_strategy: str
    processing_time_ms: float
    coherence_score: float
    factuality_score: float
    issues_count: int
    diagnostic_details: Dict[str, Any]


@dataclass
class MockLevel3Result:
    """Résultat simulé niveau 3."""
    improvement_successful: bool
    is_accepted: bool
    final_confidence: float
    strategy_applied: str
    improvements_made: List[str]
    processing_time_ms: float
    initial_quality_score: float
    final_quality_score: float
    iterations_count: int
    timed_out: bool
    escalated: bool


def generate_mock_level0_results(count: int = 100) -> List[MockFilterResult]:
    """Génère résultats simulés niveau 0 avec variété réaliste."""
    
    results = []
    
    for i in range(count):
        # Simulation distribution réaliste
        corruption_score = np.random.exponential(0.05)  # Distribution long-tail
        is_valid = corruption_score < 0.1
        
        corrections = []
        rejection_reasons = []
        
        if corruption_score > 0.15:
            rejection_reasons.append("Corruption excessive")
        if np.random.random() < 0.2:  # 20% ont des corrections
            corrections.append("removed_repetitions")
        if np.random.random() < 0.1:  # 10% corruption confidence_weighted
            corrections.append("fix_confidence_weighted_corruption")
            corruption_score += 0.3
        
        # Temps traitement réaliste
        processing_time = np.random.normal(150, 50)  # 150ms ± 50ms
        
        # Longueurs
        original_words = int(np.random.normal(200, 100))
        word_reduction = int(np.random.normal(20, 10)) if corrections else 0
        final_words = max(10, original_words - word_reduction)
        
        results.append(MockFilterResult(
            is_valid=is_valid,
            can_be_used=is_valid or len(corrections) > 0,
            processing_time_ms=max(50, processing_time),
            corrections_applied=corrections,
            rejection_reasons=rejection_reasons,
            corruption_score=min(1.0, corruption_score),
            original_word_count=max(10, original_words),
            word_count=final_words
        ))
    
    return results


def generate_mock_level1_results(count: int = 100) -> List[MockHeuristicResult]:
    """Génère résultats simulés niveau 1."""
    
    results = []
    
    for i in range(count):
        # Distribution confiance
        confidence = np.random.beta(2, 1)  # Biais vers confiance élevée
        is_suspect = confidence < 0.6
        
        # Niveau risque selon confiance
        if confidence > 0.8:
            risk_level = "low"
        elif confidence > 0.6:
            risk_level = "medium"
        elif confidence > 0.4:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Issues simulées
        issues = []
        if confidence < 0.7:
            issues.append({
                'type': 'coherence_faible',
                'severity': 'moderate' if confidence > 0.5 else 'critical'
            })
        if np.random.random() < 0.1:  # 10% corruption confidence_weighted
            issues.append({
                'type': 'corruption_confidence_weighted',
                'severity': 'critical'
            })
        
        # Entités et fact-checking
        entities = int(np.random.poisson(3))  # Moyenne 3 entités
        fact_checks = [{'type': 'entity_verification'} for _ in range(min(2, entities))]
        
        results.append(MockHeuristicResult(
            is_suspect=is_suspect,
            confidence_score=confidence,
            risk_level=risk_level,
            issues=issues,
            processing_time_ms=np.random.normal(200, 75),
            word_count=int(np.random.normal(80, 30)),
            entities_detected=entities,
            fact_check_candidates=fact_checks
        ))
    
    return results


def generate_mock_level2_results(count: int = 100) -> List[MockLevel2Result]:
    """Génère résultats simulés niveau 2 avec classification intelligente."""
    
    # Enum simulé pour tier classification
    class MockTier:
        def __init__(self, value):
            self.value = value
    
    results = []
    
    for i in range(count):
        # Distribution tier réaliste
        rand = np.random.random()
        if rand < 0.2:
            tier = MockTier("EXCELLENT")
            confidence = np.random.uniform(0.8, 1.0)
            coherence = np.random.uniform(0.8, 1.0)
            factuality = np.random.uniform(0.9, 1.0)
            strategy = "none"
        elif rand < 0.4:
            tier = MockTier("GOOD")
            confidence = np.random.uniform(0.65, 0.85)
            coherence = np.random.uniform(0.7, 0.85)
            factuality = np.random.uniform(0.8, 0.9)
            strategy = "edit" if np.random.random() < 0.3 else "none"
        elif rand < 0.7:
            tier = MockTier("MODERATE")
            confidence = np.random.uniform(0.5, 0.7)
            coherence = np.random.uniform(0.5, 0.7)
            factuality = np.random.uniform(0.7, 0.85)
            strategy = "edit"
        elif rand < 0.85:
            tier = MockTier("CRITICAL_RECOVERABLE")
            confidence = np.random.uniform(0.3, 0.5)
            coherence = np.random.uniform(0.4, 0.6)
            factuality = np.random.uniform(0.6, 0.8)
            strategy = "edit"
        elif rand < 0.95:
            tier = MockTier("CRITICAL_HALLUCINATION")
            confidence = np.random.uniform(0.05, 0.15)
            coherence = np.random.uniform(0.1, 0.3)
            factuality = np.random.uniform(0.1, 0.4)
            strategy = "regenerate"
        else:
            tier = MockTier("CRITICAL_CORRUPTED")
            confidence = np.random.uniform(0.0, 0.1)
            coherence = np.random.uniform(0.0, 0.2)
            factuality = np.random.uniform(0.0, 0.3)
            strategy = "escalate"
        
        # Validation selon tier
        is_valid = tier.value in ["EXCELLENT", "GOOD", "MODERATE"]
        is_production_ready = tier.value in ["EXCELLENT", "GOOD"]
        
        # Diagnostic
        diagnostic = {
            'has_hallucination': tier.value == "CRITICAL_HALLUCINATION",
            'has_corruption': tier.value == "CRITICAL_CORRUPTED"
        }
        
        results.append(MockLevel2Result(
            is_valid=is_valid,
            is_production_ready=is_production_ready,
            validation_confidence=confidence,
            tier_classification=tier,
            level3_strategy=strategy,
            processing_time_ms=np.random.normal(300, 100),
            coherence_score=coherence,
            factuality_score=factuality,
            issues_count=int(np.random.poisson(2)),
            diagnostic_details=diagnostic
        ))
    
    return results


def generate_mock_level3_results(count: int = 100) -> List[MockLevel3Result]:
    """Génère résultats simulés niveau 3 avec améliorations adaptatives."""
    
    results = []
    
    for i in range(count):
        # Stratégies et succès
        strategies = ["edit_intelligent", "regenerate_from_source", "escalate_manual", "bypass_acceptable"]
        strategy = np.random.choice(strategies)
        
        # Taux succès selon stratégie
        if strategy == "edit_intelligent":
            success_rate = 0.75
            improvement_range = (0.1, 0.3)
        elif strategy == "regenerate_from_source":
            success_rate = 0.85
            improvement_range = (0.3, 0.6)
        elif strategy == "escalate_manual":
            success_rate = 0.95
            improvement_range = (0.5, 0.8)
        else:  # bypass
            success_rate = 1.0
            improvement_range = (0.0, 0.05)
        
        is_successful = np.random.random() < success_rate
        is_accepted = is_successful
        
        # Scores qualité
        initial_score = np.random.uniform(0.2, 0.6)
        if is_successful:
            improvement = np.random.uniform(*improvement_range)
            final_score = min(1.0, initial_score + improvement)
        else:
            final_score = initial_score * np.random.uniform(0.8, 1.0)  # Légère dégradation possible
        
        # Améliorations
        improvements = []
        if is_successful and strategy == "edit_intelligent":
            improvements.extend(["coherence_improved", "repetitions_removed"])
        elif is_successful and strategy == "regenerate_from_source":
            improvements.extend(["hallucination_fixed", "topic_aligned"])
        
        # Temps traitement (plus long pour niveau 3)
        base_time = 500 if strategy == "regenerate_from_source" else 200
        processing_time = np.random.normal(base_time, base_time * 0.3)
        
        results.append(MockLevel3Result(
            improvement_successful=is_successful,
            is_accepted=is_accepted,
            final_confidence=final_score,
            strategy_applied=strategy,
            improvements_made=improvements,
            processing_time_ms=max(100, processing_time),
            initial_quality_score=initial_score,
            final_quality_score=final_score,
            iterations_count=int(np.random.poisson(2)) + 1,
            timed_out=not is_successful and np.random.random() < 0.1,
            escalated=strategy == "escalate_manual"
        ))
    
    return results


def test_pipeline_evaluation():
    """Test complet du système d'évaluation pipeline."""
    
    print("=== TEST SYSTÈME ÉVALUATION PIPELINE COMPLET ===\n")
    
    # Génération données test
    print("1. Génération données test...")
    level0_results = generate_mock_level0_results(100)
    level1_results = generate_mock_level1_results(95)  # Quelques rejets niveau 0
    level2_results = generate_mock_level2_results(90)  # Quelques rejets niveau 1
    level3_results = generate_mock_level3_results(45)  # Seulement critiques/modérés traités
    
    # Input data simulé
    input_data = {
        'summaries': {f'article_{i}': {} for i in range(100)},
        'articles': [{'id': i} for i in range(100)]
    }
    
    print(f"  - Niveau 0: {len(level0_results)} résultats")
    print(f"  - Niveau 1: {len(level1_results)} résultats")
    print(f"  - Niveau 2: {len(level2_results)} résultats")
    print(f"  - Niveau 3: {len(level3_results)} résultats")
    print()
    
    # Création évaluateur
    print("2. Création évaluateur...")
    evaluator = create_pipeline_evaluator()
    print("✓ Évaluateur initialisé")
    print()
    
    # Évaluation complète
    print("3. Évaluation pipeline complet...")
    start_time = time.time()
    
    evaluation = evaluator.evaluate_complete_pipeline(
        input_data=input_data,
        level0_results=level0_results,
        level1_results=level1_results,
        level2_results=level2_results,
        level3_results=level3_results
    )
    
    evaluation_time = (time.time() - start_time) * 1000
    print(f"✓ Évaluation terminée en {evaluation_time:.1f}ms")
    print()
    
    # Affichage résultats
    print("4. RÉSULTATS ÉVALUATION")
    print("=" * 50)
    
    # Performance globale
    print("📊 PERFORMANCE GLOBALE:")
    overall = evaluation.overall_performance
    print(f"  - Score global: {overall['overall_score']:.3f}")
    print(f"  - Précision moyenne: {overall['avg_accuracy']:.3f}")
    print(f"  - Confiance moyenne: {overall['avg_confidence']:.3f}")
    print(f"  - Temps total: {overall['total_processing_time_ms']:.1f}ms")
    print(f"  - Débit: {overall['pipeline_throughput']:.1f} échantillons/sec")
    print()
    
    # Performance par niveau
    print("🎯 PERFORMANCE PAR NIVEAU:")
    for level, metrics in evaluation.level_metrics.items():
        print(f"  Niveau {level}:")
        print(f"    - Traités: {metrics.total_processed}")
        print(f"    - Valides: {metrics.valid_outputs} ({metrics.validation_rate:.1f}%)")
        print(f"    - Confiance: {metrics.avg_confidence:.3f}")
        print(f"    - Temps: {metrics.processing_time_ms:.1f}ms")
        
        # Détails spécifiques par niveau
        if level == 0:
            perf = metrics.performance_details
            print(f"    - Corrections: {perf.get('correction_rate', 0):.1f}%")
            print(f"    - Utilisabilité: {perf.get('usability_rate', 0):.1f}%")
        elif level == 1:
            perf = metrics.performance_details
            print(f"    - Suspects: {perf.get('suspect_rate', 0):.1f}%")
            print(f"    - Entités moy: {perf.get('avg_entities_detected', 0):.1f}")
        elif level == 2:
            perf = metrics.performance_details
            print(f"    - Production: {perf.get('production_ready_rate', 0):.1f}%")
            print(f"    - Hallucinations: {perf.get('hallucination_rate', 0):.1f}%")
        elif level == 3:
            perf = metrics.performance_details
            print(f"    - Améliorations: {perf.get('improvement_success_rate', 0):.1f}%")
            print(f"    - Acceptation: {perf.get('acceptance_rate', 0):.1f}%")
        print()
    
    # Robustesse
    print("🛡️ ROBUSTESSE:")
    robustness = evaluation.robustness_assessment
    print(f"  - Gestion corruption: {robustness['corruption_handling']:.3f}")
    print(f"  - Détection hallucination: {robustness['hallucination_detection']:.3f}")
    print(f"  - Consistance performance: {robustness['performance_consistency']:.3f}")
    print(f"  - Score global: {robustness['overall_robustness']:.3f}")
    print()
    
    # Recommandations
    print("💡 RECOMMANDATIONS:")
    for i, rec in enumerate(evaluation.recommendations, 1):
        print(f"  {i}. {rec}")
    print()
    
    # Test export
    print("5. Test export rapport...")
    output_path = Path("evaluation_report_test.json")
    evaluator.export_evaluation_report(evaluation, output_path)
    
    if output_path.exists():
        print(f"✓ Rapport exporté: {output_path}")
        file_size = output_path.stat().st_size
        print(f"  Taille: {file_size} bytes")
        
        # Validation JSON
        try:
            with open(output_path, 'r') as f:
                json.load(f)
            print("✓ JSON valide")
        except json.JSONDecodeError as e:
            print(f"✗ Erreur JSON: {e}")
    else:
        print("✗ Échec export")
    print()
    
    # Analyse qualitative
    print("6. ANALYSE QUALITATIVE")
    print("=" * 50)
    
    # Goulots d'étranglement
    bottlenecks = []
    for level, metrics in evaluation.level_metrics.items():
        if metrics.validation_rate < 70:
            bottlenecks.append(f"Niveau {level} (validation {metrics.validation_rate:.1f}%)")
        if metrics.processing_time_ms > 400:
            bottlenecks.append(f"Niveau {level} (temps {metrics.processing_time_ms:.1f}ms)")
    
    if bottlenecks:
        print("⚠️ Goulots d'étranglement détectés:")
        for bottleneck in bottlenecks:
            print(f"  - {bottleneck}")
    else:
        print("✅ Aucun goulot d'étranglement majeur")
    print()
    
    # Points forts
    strengths = []
    for level, metrics in evaluation.level_metrics.items():
        if metrics.validation_rate > 85:
            strengths.append(f"Niveau {level}: Excellent taux validation ({metrics.validation_rate:.1f}%)")
        if metrics.avg_confidence > 0.8:
            strengths.append(f"Niveau {level}: Haute confiance ({metrics.avg_confidence:.3f})")
    
    if strengths:
        print("💪 Points forts:")
        for strength in strengths:
            print(f"  - {strength}")
    print()
    
    # Score final et évaluation
    final_score = overall['overall_score']
    if final_score >= 0.8:
        assessment = "EXCELLENT - Pipeline prêt pour production"
    elif final_score >= 0.7:
        assessment = "BON - Quelques optimisations recommandées"
    elif final_score >= 0.6:
        assessment = "ACCEPTABLE - Améliorations nécessaires"
    else:
        assessment = "INSUFFISANT - Révision majeure requise"
    
    print(f"🎯 ÉVALUATION FINALE: {assessment}")
    print(f"   Score: {final_score:.3f}")
    print()
    
    print("✅ Test système évaluation terminé avec succès!")
    
    # Cleanup
    if output_path.exists():
        output_path.unlink()


if __name__ == "__main__":
    try:
        test_pipeline_evaluation()
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
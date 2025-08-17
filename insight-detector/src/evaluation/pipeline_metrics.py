# src/evaluation/pipeline_metrics.py
"""
Métriques d'évaluation robustes pour le pipeline InsightDetector complet.

Évalue la performance de chaque niveau (0-3) et du pipeline global avec:
- Métriques de détection (précision, rappel, F1)
- Métriques de qualité (amélioration, dégradation)
- Métriques de robustesse (corruption, hallucination)
- Analyses de performance comparative
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class LevelMetrics:
    """Métriques pour un niveau du pipeline."""
    level: int
    total_processed: int
    valid_outputs: int
    invalid_outputs: int
    processing_time_ms: float
    avg_confidence: float
    validation_rate: float
    issues_detected: Dict[str, int]
    quality_distribution: Dict[str, int]
    performance_details: Dict[str, Any]


@dataclass
class PipelineEvaluation:
    """Évaluation complète du pipeline."""
    pipeline_version: str
    evaluation_timestamp: str
    total_samples: int
    level_metrics: Dict[int, LevelMetrics]
    overall_performance: Dict[str, Any]
    quality_improvements: Dict[str, Any]
    robustness_assessment: Dict[str, Any]
    comparative_analysis: Dict[str, Any]
    recommendations: List[str]


class PipelineEvaluator:
    """
    Évaluateur robuste pour le pipeline InsightDetector complet.
    
    Fonctionnalités:
    - Évaluation performance par niveau (0-3)
    - Métriques de qualité bout-en-bout
    - Détection problèmes spécifiques
    - Comparaison avec baselines
    - Recommandations d'amélioration
    """
    
    def __init__(self, 
                 enable_detailed_analysis: bool = True,
                 baseline_comparison: bool = True,
                 quality_thresholds: Optional[Dict] = None):
        """
        Initialise l'évaluateur pipeline.
        
        Args:
            enable_detailed_analysis: Active analyse détaillée par composant
            baseline_comparison: Active comparaison avec baselines
            quality_thresholds: Seuils qualité personnalisés
        """
        
        self.enable_detailed_analysis = enable_detailed_analysis
        self.baseline_comparison = baseline_comparison
        
        # Seuils qualité par défaut (calibrés sur analyses empiriques)
        self.quality_thresholds = quality_thresholds or {
            'excellent_confidence': 0.80,
            'good_confidence': 0.65,
            'acceptable_confidence': 0.50,
            'min_validation_rate': 0.70,
            'max_processing_time_ms': 5000,
            'min_improvement_rate': 0.60,
            'max_corruption_rate': 0.05,
            'max_hallucination_rate': 0.02
        }
        
        # Poids métriques pour score composite
        self.metric_weights = {
            'accuracy': 0.25,
            'quality_improvement': 0.25,
            'robustness': 0.20,
            'efficiency': 0.15,
            'usability': 0.15
        }
        
        # Cache résultats pour performance
        self.evaluation_cache = {}
        
    def evaluate_complete_pipeline(self, 
                                 input_data: Dict[str, Any],
                                 level0_results: List[Any] = None,
                                 level1_results: List[Any] = None,
                                 level2_results: List[Any] = None,
                                 level3_results: List[Any] = None,
                                 baseline_results: Optional[Dict] = None) -> PipelineEvaluation:
        """
        Évaluation complète du pipeline avec tous les niveaux.
        
        Args:
            input_data: Données d'entrée (articles + résumés)
            level0_results: Résultats du préfiltre
            level1_results: Résultats heuristique
            level2_results: Résultats classification
            level3_results: Résultats amélioration
            baseline_results: Résultats baseline pour comparaison
            
        Returns:
            PipelineEvaluation avec métriques complètes
        """
        
        start_time = time.time()
        
        # Métadonnées évaluation
        evaluation_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        pipeline_version = "enhanced_v1.0"
        total_samples = len(input_data.get('summaries', {}))
        
        # Évaluation par niveau
        level_metrics = {}
        
        if level0_results:
            level_metrics[0] = self._evaluate_level0(level0_results, input_data)
        
        if level1_results:
            level_metrics[1] = self._evaluate_level1(level1_results, input_data)
        
        if level2_results:
            level_metrics[2] = self._evaluate_level2(level2_results, input_data)
        
        if level3_results:
            level_metrics[3] = self._evaluate_level3(level3_results, input_data)
        
        # Performance globale
        overall_performance = self._calculate_overall_performance(level_metrics, total_samples)
        
        # Amélioration qualité
        quality_improvements = self._assess_quality_improvements(
            input_data, level0_results, level1_results, level2_results, level3_results
        )
        
        # Robustesse
        robustness_assessment = self._assess_robustness(level_metrics, input_data)
        
        # Analyse comparative
        comparative_analysis = {}
        if baseline_results and self.baseline_comparison:
            comparative_analysis = self._compare_with_baseline(
                level_metrics, baseline_results
            )
        
        # Recommandations
        recommendations = self._generate_recommendations(
            level_metrics, overall_performance, quality_improvements, robustness_assessment
        )
        
        evaluation_time = (time.time() - start_time) * 1000
        logger.info(f"Évaluation pipeline terminée en {evaluation_time:.1f}ms")
        
        return PipelineEvaluation(
            pipeline_version=pipeline_version,
            evaluation_timestamp=evaluation_timestamp,
            total_samples=total_samples,
            level_metrics=level_metrics,
            overall_performance=overall_performance,
            quality_improvements=quality_improvements,
            robustness_assessment=robustness_assessment,
            comparative_analysis=comparative_analysis,
            recommendations=recommendations
        )
    
    def _evaluate_level0(self, results: List[Any], input_data: Dict) -> LevelMetrics:
        """Évaluation spécifique niveau 0 (préfiltre)."""
        
        total_processed = len(results)
        valid_outputs = sum(1 for r in results if getattr(r, 'is_valid', False))
        invalid_outputs = total_processed - valid_outputs
        
        # Temps traitement
        processing_times = [getattr(r, 'processing_time_ms', 0) for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Confiance moyenne (pour résultats utilisant can_be_used)
        usable_results = [r for r in results if getattr(r, 'can_be_used', False)]
        avg_confidence = len(usable_results) / total_processed if total_processed > 0 else 0
        
        validation_rate = valid_outputs / total_processed * 100 if total_processed > 0 else 0
        
        # Issues détectées
        issues_detected = defaultdict(int)
        for result in results:
            rejection_reasons = getattr(result, 'rejection_reasons', [])
            for reason in rejection_reasons:
                issues_detected[reason] += 1
        
        # Distribution correction
        corrections_applied = defaultdict(int)
        for result in results:
            corrections = getattr(result, 'corrections_applied', [])
            for correction in corrections:
                corrections_applied[correction] += 1
        
        # Détails performance
        performance_details = {
            'avg_processing_time_ms': avg_processing_time,
            'correction_rate': sum(1 for r in results if getattr(r, 'corrections_applied', [])) / total_processed * 100,
            'usability_rate': len(usable_results) / total_processed * 100,
            'top_rejection_reasons': dict(Counter(issues_detected).most_common(5)),
            'top_corrections_applied': dict(Counter(corrections_applied).most_common(5)),
            'avg_corruption_score': np.mean([getattr(r, 'corruption_score', 0) for r in results]),
            'avg_word_count_reduction': np.mean([
                getattr(r, 'original_word_count', 0) - getattr(r, 'word_count', 0) 
                for r in results
            ])
        }
        
        return LevelMetrics(
            level=0,
            total_processed=total_processed,
            valid_outputs=valid_outputs,
            invalid_outputs=invalid_outputs,
            processing_time_ms=avg_processing_time,
            avg_confidence=avg_confidence,
            validation_rate=validation_rate,
            issues_detected=dict(issues_detected),
            quality_distribution=dict(corrections_applied),
            performance_details=performance_details
        )
    
    def _evaluate_level1(self, results: List[Any], input_data: Dict) -> LevelMetrics:
        """Évaluation spécifique niveau 1 (heuristique)."""
        
        total_processed = len(results)
        
        # Suspects détectés
        suspects = sum(1 for r in results if getattr(r, 'is_suspect', False))
        non_suspects = total_processed - suspects
        
        # Métriques confiance
        confidence_scores = [getattr(r, 'confidence_score', 0) for r in results]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Taux détection (inversé car on veut des non-suspects)
        validation_rate = non_suspects / total_processed * 100 if total_processed > 0 else 0
        
        # Issues par type et sévérité
        issues_detected = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for result in results:
            issues = getattr(result, 'issues', [])
            for issue in issues:
                issue_type = issue.get('type', 'unknown')
                severity = issue.get('severity', 'minor')
                issues_detected[issue_type] += 1
                severity_counts[severity] += 1
        
        # Distribution risque
        risk_levels = [getattr(r, 'risk_level', 'low') for r in results]
        risk_distribution = dict(Counter(risk_levels))
        
        # Temps traitement
        processing_times = [getattr(r, 'processing_time_ms', 0) for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        performance_details = {
            'avg_processing_time_ms': avg_processing_time,
            'suspect_rate': suspects / total_processed * 100,
            'avg_confidence': avg_confidence,
            'confidence_distribution': {
                'high': sum(1 for c in confidence_scores if c >= 0.8),
                'medium': sum(1 for c in confidence_scores if 0.5 <= c < 0.8),
                'low': sum(1 for c in confidence_scores if c < 0.5)
            },
            'risk_distribution': risk_distribution,
            'severity_distribution': dict(severity_counts),
            'top_issue_types': dict(Counter(issues_detected).most_common(10)),
            'avg_word_count': np.mean([getattr(r, 'word_count', 0) for r in results]),
            'avg_entities_detected': np.mean([getattr(r, 'entities_detected', 0) for r in results]),
            'fact_check_candidates_rate': np.mean([
                len(getattr(r, 'fact_check_candidates', [])) for r in results
            ])
        }
        
        return LevelMetrics(
            level=1,
            total_processed=total_processed,
            valid_outputs=non_suspects,
            invalid_outputs=suspects,
            processing_time_ms=avg_processing_time,
            avg_confidence=avg_confidence,
            validation_rate=validation_rate,
            issues_detected=dict(issues_detected),
            quality_distribution=risk_distribution,
            performance_details=performance_details
        )
    
    def _evaluate_level2(self, results: List[Any], input_data: Dict) -> LevelMetrics:
        """Évaluation spécifique niveau 2 (classification intelligente)."""
        
        total_processed = len(results)
        
        # Validation selon niveau 2
        valid_outputs = sum(1 for r in results if getattr(r, 'is_valid', False))
        production_ready = sum(1 for r in results if getattr(r, 'is_production_ready', False))
        
        # Confiance moyenne
        confidence_scores = [getattr(r, 'validation_confidence', 0) for r in results]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        validation_rate = valid_outputs / total_processed * 100 if total_processed > 0 else 0
        
        # Distribution par tier (classification intelligente)
        tier_counts = defaultdict(int)
        strategy_counts = defaultdict(int)
        
        for result in results:
            tier = getattr(result, 'tier_classification', None)
            if tier:
                tier_name = getattr(tier, 'value', str(tier))
                tier_counts[tier_name] += 1
            
            strategy = getattr(result, 'level3_strategy', 'unknown')
            strategy_counts[strategy] += 1
        
        # Problèmes détectés
        issues_detected = defaultdict(int)
        for result in results:
            diagnostic = getattr(result, 'diagnostic_details', {})
            if diagnostic.get('has_hallucination'):
                issues_detected['hallucination'] += 1
            if diagnostic.get('has_corruption'):
                issues_detected['corruption'] += 1
        
        # Temps traitement
        processing_times = [getattr(r, 'processing_time_ms', 0) for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        performance_details = {
            'avg_processing_time_ms': avg_processing_time,
            'production_ready_rate': production_ready / total_processed * 100,
            'avg_confidence': avg_confidence,
            'tier_distribution': dict(tier_counts),
            'level3_strategy_distribution': dict(strategy_counts),
            'problems_detected': dict(issues_detected),
            'avg_coherence': np.mean([getattr(r, 'coherence_score', 0) for r in results]),
            'avg_factuality': np.mean([getattr(r, 'factuality_score', 0) for r in results]),
            'avg_issues_count': np.mean([getattr(r, 'issues_count', 0) for r in results]),
            'hallucination_rate': issues_detected['hallucination'] / total_processed * 100,
            'corruption_rate': issues_detected['corruption'] / total_processed * 100
        }
        
        return LevelMetrics(
            level=2,
            total_processed=total_processed,
            valid_outputs=valid_outputs,
            invalid_outputs=total_processed - valid_outputs,
            processing_time_ms=avg_processing_time,
            avg_confidence=avg_confidence,
            validation_rate=validation_rate,
            issues_detected=dict(issues_detected),
            quality_distribution=dict(tier_counts),
            performance_details=performance_details
        )
    
    def _evaluate_level3(self, results: List[Any], input_data: Dict) -> LevelMetrics:
        """Évaluation spécifique niveau 3 (amélioration adaptative)."""
        
        total_processed = len(results)
        
        # Succès amélioration
        successful_improvements = sum(1 for r in results if getattr(r, 'improvement_successful', False))
        accepted_outputs = sum(1 for r in results if getattr(r, 'is_accepted', False))
        
        # Confiance finale
        confidence_scores = [getattr(r, 'final_confidence', 0) for r in results]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        validation_rate = accepted_outputs / total_processed * 100 if total_processed > 0 else 0
        
        # Stratégies utilisées
        strategies_used = defaultdict(int)
        for result in results:
            strategy = getattr(result, 'strategy_applied', 'unknown')
            strategies_used[strategy] += 1
        
        # Améliorations mesurées
        improvements = defaultdict(int)
        for result in results:
            improvement_types = getattr(result, 'improvements_made', [])
            for improvement in improvement_types:
                improvements[improvement] += 1
        
        # Temps traitement (critique pour niveau 3)
        processing_times = [getattr(r, 'processing_time_ms', 0) for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Calcul amélioration qualité
        quality_improvements = []
        for result in results:
            initial_score = getattr(result, 'initial_quality_score', 0)
            final_score = getattr(result, 'final_quality_score', 0)
            if initial_score > 0:
                improvement = (final_score - initial_score) / initial_score
                quality_improvements.append(improvement)
        
        avg_quality_improvement = np.mean(quality_improvements) if quality_improvements else 0
        
        performance_details = {
            'avg_processing_time_ms': avg_processing_time,
            'improvement_success_rate': successful_improvements / total_processed * 100,
            'acceptance_rate': accepted_outputs / total_processed * 100,
            'avg_quality_improvement': avg_quality_improvement,
            'strategies_distribution': dict(strategies_used),
            'improvements_distribution': dict(improvements),
            'timeout_rate': sum(1 for r in results if getattr(r, 'timed_out', False)) / total_processed * 100,
            'escalation_rate': sum(1 for r in results if getattr(r, 'escalated', False)) / total_processed * 100,
            'avg_iterations': np.mean([getattr(r, 'iterations_count', 1) for r in results]),
            'quality_improvement_distribution': {
                'significant': sum(1 for qi in quality_improvements if qi > 0.2),
                'moderate': sum(1 for qi in quality_improvements if 0.05 <= qi <= 0.2),
                'minimal': sum(1 for qi in quality_improvements if 0 <= qi < 0.05),
                'degraded': sum(1 for qi in quality_improvements if qi < 0)
            }
        }
        
        return LevelMetrics(
            level=3,
            total_processed=total_processed,
            valid_outputs=accepted_outputs,
            invalid_outputs=total_processed - accepted_outputs,
            processing_time_ms=avg_processing_time,
            avg_confidence=avg_confidence,
            validation_rate=validation_rate,
            issues_detected={},
            quality_distribution=dict(strategies_used),
            performance_details=performance_details
        )
    
    def _calculate_overall_performance(self, level_metrics: Dict[int, LevelMetrics], 
                                     total_samples: int) -> Dict[str, Any]:
        """Calcul performance globale pipeline."""
        
        # Métriques bout-en-bout
        total_processing_time = sum(
            metrics.processing_time_ms for metrics in level_metrics.values()
        )
        
        # Taux passage entre niveaux
        level_progression = {}
        for level in sorted(level_metrics.keys()):
            if level > 0 and (level - 1) in level_metrics:
                prev_valid = level_metrics[level - 1].valid_outputs
                curr_processed = level_metrics[level].total_processed
                progression_rate = curr_processed / prev_valid * 100 if prev_valid > 0 else 0
                level_progression[f"level_{level-1}_to_{level}"] = progression_rate
        
        # Score composite
        accuracy_scores = [m.validation_rate / 100 for m in level_metrics.values()]
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
        
        confidence_scores = [m.avg_confidence for m in level_metrics.values()]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Efficacité (vitesse vs qualité)
        efficiency_score = self._calculate_efficiency_score(level_metrics)
        
        # Score global pondéré
        overall_score = (
            avg_accuracy * self.metric_weights['accuracy'] +
            avg_confidence * self.metric_weights['quality_improvement'] +
            efficiency_score * self.metric_weights['efficiency']
        )
        
        return {
            'overall_score': overall_score,
            'avg_accuracy': avg_accuracy,
            'avg_confidence': avg_confidence,
            'total_processing_time_ms': total_processing_time,
            'avg_processing_time_per_sample': total_processing_time / total_samples if total_samples > 0 else 0,
            'level_progression_rates': level_progression,
            'efficiency_score': efficiency_score,
            'pipeline_throughput': total_samples / (total_processing_time / 1000) if total_processing_time > 0 else 0,
            'quality_vs_speed_ratio': avg_accuracy / (total_processing_time / 1000) if total_processing_time > 0 else 0
        }
    
    def _assess_quality_improvements(self, input_data: Dict, *level_results) -> Dict[str, Any]:
        """Évaluation amélioration qualité bout-en-bout."""
        
        # Comparaison qualité initiale vs finale
        improvements = {
            'corruption_reduction': 0,
            'hallucination_reduction': 0,
            'coherence_improvement': 0,
            'factuality_improvement': 0,
            'overall_improvement': 0
        }
        
        # Si données disponibles, calculer améliorations réelles
        if len(level_results) > 1:
            initial_results = level_results[0] if level_results[0] else []
            final_results = level_results[-1] if level_results[-1] else []
            
            if initial_results and final_results:
                # Calcul améliorations spécifiques
                improvements['corruption_reduction'] = self._calculate_corruption_reduction(
                    initial_results, final_results
                )
                improvements['hallucination_reduction'] = self._calculate_hallucination_reduction(
                    initial_results, final_results
                )
        
        return improvements
    
    def _assess_robustness(self, level_metrics: Dict[int, LevelMetrics], 
                         input_data: Dict) -> Dict[str, Any]:
        """Évaluation robustesse pipeline."""
        
        robustness = {
            'corruption_handling': 0,
            'hallucination_detection': 0,
            'error_recovery': 0,
            'performance_consistency': 0,
            'overall_robustness': 0
        }
        
        # Analyse corruption handling (niveau 0-1)
        if 0 in level_metrics and 1 in level_metrics:
            level0_perf = level_metrics[0].performance_details
            corruption_rate = level0_perf.get('avg_corruption_score', 0)
            correction_rate = level0_perf.get('correction_rate', 0)
            
            robustness['corruption_handling'] = max(0, 1 - corruption_rate) * (correction_rate / 100)
        
        # Analyse hallucination detection (niveau 2)
        if 2 in level_metrics:
            level2_perf = level_metrics[2].performance_details
            hallucination_rate = level2_perf.get('hallucination_rate', 0) / 100
            robustness['hallucination_detection'] = max(0, 1 - hallucination_rate * 10)  # Pénalité forte
        
        # Consistance performance
        validation_rates = [m.validation_rate for m in level_metrics.values()]
        if validation_rates:
            consistency = 1 - (np.std(validation_rates) / np.mean(validation_rates))
            robustness['performance_consistency'] = max(0, consistency)
        
        # Score global robustesse
        robustness['overall_robustness'] = np.mean([
            robustness['corruption_handling'],
            robustness['hallucination_detection'],
            robustness['performance_consistency']
        ])
        
        return robustness
    
    def _generate_recommendations(self, level_metrics: Dict[int, LevelMetrics],
                                overall_performance: Dict[str, Any],
                                quality_improvements: Dict[str, Any],
                                robustness_assessment: Dict[str, Any]) -> List[str]:
        """Génération recommandations d'amélioration."""
        
        recommendations = []
        
        # Recommandations par niveau
        for level, metrics in level_metrics.items():
            if metrics.validation_rate < self.quality_thresholds['min_validation_rate'] * 100:
                recommendations.append(
                    f"Niveau {level}: Taux validation faible ({metrics.validation_rate:.1f}%) - "
                    f"revoir seuils et algorithmes"
                )
            
            if metrics.processing_time_ms > self.quality_thresholds['max_processing_time_ms']:
                recommendations.append(
                    f"Niveau {level}: Temps traitement excessif ({metrics.processing_time_ms:.1f}ms) - "
                    f"optimiser performance"
                )
        
        # Recommandations robustesse
        if robustness_assessment['corruption_handling'] < 0.7:
            recommendations.append(
                "Améliorer gestion corruption - renforcer préfiltre et validation"
            )
        
        if robustness_assessment['hallucination_detection'] < 0.8:
            recommendations.append(
                "Renforcer détection hallucinations - améliorer cohérence thématique"
            )
        
        # Recommandations performance globale
        if overall_performance['overall_score'] < 0.7:
            recommendations.append(
                "Score global faible - révision complète pipeline recommandée"
            )
        
        if overall_performance['efficiency_score'] < 0.6:
            recommendations.append(
                "Optimiser ratio qualité/vitesse - profiler et optimiser composants lents"
            )
        
        # Recommandations spécifiques issues
        for level, metrics in level_metrics.items():
            top_issues = metrics.performance_details.get('top_issue_types', {})
            if 'corruption_confidence_weighted' in top_issues:
                recommendations.append(
                    "Problème critique confidence_weighted détecté - remplacer par alternative"
                )
        
        return recommendations
    
    def _calculate_efficiency_score(self, level_metrics: Dict[int, LevelMetrics]) -> float:
        """Calcul score efficacité (qualité vs vitesse)."""
        
        if not level_metrics:
            return 0.0
        
        # Normalisation temps traitement (log scale)
        processing_times = [m.processing_time_ms for m in level_metrics.values()]
        avg_time = np.mean(processing_times)
        time_score = max(0, 1 - np.log10(avg_time / 100) / 2)  # Pénalité logarithmique
        
        # Score qualité
        validation_rates = [m.validation_rate / 100 for m in level_metrics.values()]
        quality_score = np.mean(validation_rates)
        
        # Efficacité = qualité * rapidité
        efficiency_score = quality_score * time_score
        
        return max(0, min(1, efficiency_score))
    
    def _calculate_corruption_reduction(self, initial_results: List, 
                                      final_results: List) -> float:
        """Calcul réduction corruption entre début et fin."""
        # Implémentation simplifiée - à adapter selon structures données
        return 0.5  # Placeholder
    
    def _calculate_hallucination_reduction(self, initial_results: List,
                                         final_results: List) -> float:
        """Calcul réduction hallucinations entre début et fin.""" 
        # Implémentation simplifiée - à adapter selon structures données
        return 0.3  # Placeholder
    
    def export_evaluation_report(self, evaluation: PipelineEvaluation, 
                               output_path: Path) -> None:
        """Export rapport évaluation en JSON."""
        
        # Conversion en dict pour JSON
        evaluation_dict = asdict(evaluation)
        
        # Ajout métadonnées
        evaluation_dict['export_metadata'] = {
            'exported_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'evaluator_version': '1.0',
            'quality_thresholds_used': self.quality_thresholds
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Rapport évaluation exporté: {output_path}")
    
    def compare_evaluations(self, evaluation1: PipelineEvaluation,
                          evaluation2: PipelineEvaluation) -> Dict[str, Any]:
        """Comparaison entre deux évaluations."""
        
        comparison = {
            'overall_score_diff': (
                evaluation2.overall_performance['overall_score'] - 
                evaluation1.overall_performance['overall_score']
            ),
            'performance_changes': {},
            'recommendations_diff': {
                'removed': list(set(evaluation1.recommendations) - set(evaluation2.recommendations)),
                'added': list(set(evaluation2.recommendations) - set(evaluation1.recommendations))
            }
        }
        
        # Comparaison par niveau
        for level in set(evaluation1.level_metrics.keys()) | set(evaluation2.level_metrics.keys()):
            if level in evaluation1.level_metrics and level in evaluation2.level_metrics:
                m1 = evaluation1.level_metrics[level]
                m2 = evaluation2.level_metrics[level]
                
                comparison['performance_changes'][f'level_{level}'] = {
                    'validation_rate_diff': m2.validation_rate - m1.validation_rate,
                    'confidence_diff': m2.avg_confidence - m1.avg_confidence,
                    'processing_time_diff': m2.processing_time_ms - m1.processing_time_ms
                }
        
        return comparison


# Fonction utilitaire
def create_pipeline_evaluator() -> PipelineEvaluator:
    """Crée un évaluateur pipeline avec configuration optimale."""
    
    return PipelineEvaluator(
        enable_detailed_analysis=True,
        baseline_comparison=True
    )
"""
Coordinateur principal du Niveau 2 - Validation factuelle adaptative.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .coherence_factuality_validator import CoherenceFactualityValidator
from .candidate_validator import CandidateValidator
from .statistical_fact_validator import StatisticalFactValidator
from .internal_consistency_analyzer import InternalConsistencyAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class Level2Result:
    """Résultat enrichi de l'analyse factuelle Niveau 2."""
    summary_id: str
    tier_classification: str  # TIER_1_SAFE, TIER_2_MODERATE, etc.
    
    # Scores de validation
    coherence_factuality_score: float  # 0-1
    candidate_validation_score: float  # 0-1
    statistical_credibility_score: float  # 0-1
    internal_consistency_score: float  # 0-1
    
    # Score composite final
    factual_confidence: float  # 0-1
    factual_risk_level: str  # 'low', 'medium', 'high', 'critical'
    
    # Métadonnées de traitement
    processing_time_ms: float
    validation_depth: str
    
    # Détails pour Niveau 3
    validation_details: Dict
    flagged_elements: List[str]
    confidence_factors: Dict
    
    def get_level3_priority(self) -> float:
        """Calcule la priorité pour le traitement Niveau 3."""
        base_priority = 1.0 - self.factual_confidence
        
        # Bonus selon les éléments flagués
        if len(self.flagged_elements) > 3:
            base_priority += 0.2
            
        # Bonus selon le tier de complexité
        tier_bonus = {
            'TIER_4_CRITICAL': 0.3,
            'TIER_3_COMPLEX': 0.2,
            'TIER_2_MODERATE': 0.1,
            'TIER_1_SAFE': 0.0
        }
        base_priority += tier_bonus.get(self.tier_classification, 0)
        
        return min(1.0, base_priority)

class Level2FactualProcessor:
    """
    Processeur principal du Niveau 2 - Validation factuelle adaptative.
    
    Traite les 372 résumés enrichis du Niveau 1 avec une approche multi-tiers
    basée sur la complexité réelle et les patterns identifiés.
    """
    
    def __init__(self, performance_mode: str = "balanced"):
        """
        Initialise le processeur de validation factuelle.
        
        Args:
            performance_mode: Mode de performance ("fast", "balanced", "thorough")
        """
        self.performance_mode = performance_mode
        
        # Initialisation des validateurs spécialisés
        self.coherence_validator = CoherenceFactualityValidator()
        self.candidate_validator = CandidateValidator()
        self.statistical_validator = StatisticalFactValidator()
        self.consistency_analyzer = InternalConsistencyAnalyzer()
        
        # Configuration des tiers de traitement
        self.tier_config = {
            'TIER_1_SAFE': {
                'time_budget_ms': 100,
                'validation_depth': 'minimal',
                'validators': ['coherence', 'consistency']
            },
            'TIER_2_MODERATE': {
                'time_budget_ms': 300,
                'validation_depth': 'targeted',
                'validators': ['coherence', 'candidate', 'consistency']
            },
            'TIER_3_COMPLEX': {
                'time_budget_ms': 700,
                'validation_depth': 'comprehensive',
                'validators': ['coherence', 'candidate', 'statistical', 'consistency']
            },
            'TIER_4_CRITICAL': {
                'time_budget_ms': 1000,
                'validation_depth': 'exhaustive',
                'validators': ['coherence', 'candidate', 'statistical', 'consistency']
            }
        }
        
        logger.info(f"Niveau 2 initialisé en mode {performance_mode}")
    
    def classify_summary_tier(self, summary_data: Dict) -> str:
        """
        Classifie un résumé dans le tier de traitement approprié.
        
        Args:
            summary_data: Données enrichies du résumé du Niveau 1
            
        Returns:
            str: Classification du tier (TIER_1_SAFE à TIER_4_CRITICAL)
        """
        grade = summary_data.get('original_grade', 'D')
        coherence = summary_data.get('coherence', 0.0)
        factuality = summary_data.get('factuality', 0.0)
        candidates_count = summary_data.get('fact_check_candidates_count', 0)
        risk_level = summary_data.get('risk_level', 'medium')
        
        # TIER 4 CRITICAL: Grades C/D uniquement (les plus problématiques)
        if grade in ['C', 'D']:
            return 'TIER_4_CRITICAL'
        
        # TIER 3 COMPLEX: Grades B+/B avec candidats nombreux ou métriques faibles
        if grade in ['B+', 'B'] and (candidates_count > 2 or coherence < 0.6):
            return 'TIER_3_COMPLEX'
        
        # TIER 2 MODERATE: Grades A+/A avec candidats ou B+/B avec peu de candidats
        if (grade in ['A+', 'A'] and candidates_count > 0) or (grade in ['B+', 'B'] and candidates_count <= 2):
            return 'TIER_2_MODERATE'
        
        # TIER 1 SAFE: Grades A+/A sans candidats et métriques élevées
        if grade in ['A+', 'A'] and candidates_count == 0 and coherence >= 0.8:
            return 'TIER_1_SAFE'
        
        # Fallback pour cas ambigus
        return 'TIER_2_MODERATE'
    
    def validate_summary(self, summary_data: Dict) -> Level2Result:
        """
        Valide un résumé selon son tier de complexité.
        
        Args:
            summary_data: Données enrichies du résumé du Niveau 1
            
        Returns:
            Level2Result: Résultat complet de la validation factuelle
        """
        start_time = time.time()
        summary_id = summary_data.get('id', 'unknown')
        
        # Classification du tier
        tier = self.classify_summary_tier(summary_data)
        tier_config = self.tier_config[tier]
        
        # Exécution des validations selon le tier
        validation_results = {}
        validators_to_run = tier_config['validators']
        
        if 'coherence' in validators_to_run:
            validation_results['coherence_factuality'] = self.coherence_validator.validate(summary_data)
        
        if 'candidate' in validators_to_run and summary_data.get('fact_check_candidates_count', 0) > 0:
            validation_results['candidate'] = self.candidate_validator.validate(summary_data)
        
        if 'statistical' in validators_to_run:
            validation_results['statistical'] = self.statistical_validator.validate(summary_data)
        
        if 'consistency' in validators_to_run:
            validation_results['consistency'] = self.consistency_analyzer.validate(summary_data)
        
        # Calcul du score composite
        factual_confidence, risk_level, flagged_elements = self._calculate_composite_score(
            validation_results, tier, summary_data
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Vérification du budget temps (avec tolérance)
        budget_ms = tier_config['time_budget_ms']
        if processing_time_ms > budget_ms * 1.5:  # Tolérance 50%
            logger.warning(f"Résumé {summary_id} dépassement budget: {processing_time_ms:.1f}ms > {budget_ms}ms")
        
        return Level2Result(
            summary_id=summary_id,
            tier_classification=tier,
            coherence_factuality_score=validation_results.get('coherence_factuality', {}).get('score', 1.0),
            candidate_validation_score=validation_results.get('candidate', {}).get('score', 1.0),
            statistical_credibility_score=validation_results.get('statistical', {}).get('score', 1.0),
            internal_consistency_score=validation_results.get('consistency', {}).get('score', 1.0),
            factual_confidence=factual_confidence,
            factual_risk_level=risk_level,
            processing_time_ms=processing_time_ms,
            validation_depth=tier_config['validation_depth'],
            validation_details=validation_results,
            flagged_elements=flagged_elements,
            confidence_factors=self._extract_confidence_factors(validation_results, summary_data)
        )
    
    def process_batch(self, enriched_summaries: List[Dict]) -> Tuple[List[Dict], List[Level2Result]]:
        """
        Traite un lot de résumés enrichis du Niveau 1.
        
        Args:
            enriched_summaries: Liste des résumés enrichis
            
        Returns:
            Tuple[valid_summaries, all_results]: Résumés validés et tous les résultats
        """
        start_time = time.time()
        valid_summaries = []
        all_results = []
        
        # Statistiques de traitement par tier
        tier_stats = {'TIER_1_SAFE': 0, 'TIER_2_MODERATE': 0, 'TIER_3_COMPLEX': 0, 'TIER_4_CRITICAL': 0}
        
        for summary_data in enriched_summaries:
            try:
                result = self.validate_summary(summary_data)
                all_results.append(result)
                
                # Comptage par tier
                tier_stats[result.tier_classification] += 1
                
                # ✅ CORRECTION CRITIQUE : Intégration production_ready du Level 1
                production_ready = summary_data.get('production_ready', True)
                original_coherence = summary_data.get('coherence', 1.0)
                
                # Logique de validation avec production_ready (Option 2)
                is_valid = (
                    production_ready and  # ✅ NOUVEAU : Respect de l'évaluation Level 1
                    result.factual_confidence > 0.5 and  # Seuil plus élevé (0.3 → 0.5)
                    result.factual_risk_level != 'critical' and
                    (original_coherence >= 0.3 or result.factual_confidence > 0.7)  # Compensation cohérence
                )
                
                if is_valid:
                    valid_summaries.append(summary_data)
                else:
                    # Log des rejets pour debugging
                    logger.info(f"Résumé {summary_data.get('id', 'unknown')} rejeté: "
                              f"production_ready={production_ready}, "
                              f"conf={result.factual_confidence:.3f}, "
                              f"risk={result.factual_risk_level}, "
                              f"coherence={original_coherence:.3f}")
                    
            except Exception as e:
                logger.error(f"Erreur traitement résumé {summary_data.get('id', 'unknown')}: {e}")
                # Résultat d'échec par défaut
                all_results.append(self._create_fallback_result(summary_data, str(e)))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(enriched_summaries) * 1000
        
        logger.info(f"Niveau 2 - Batch traité: {len(valid_summaries)}/{len(enriched_summaries)} valides")
        logger.info(f"Temps moyen: {avg_time:.1f}ms, Distribution tiers: {tier_stats}")
        
        return valid_summaries, all_results
    
    def _calculate_composite_score(self, validation_results: Dict, tier: str, 
                                  summary_data: Dict) -> Tuple[float, str, List[str]]:
        """Calcule le score composite et le niveau de risque."""
        
        # Pondération adaptative selon le tier
        weights = {
            'TIER_1_SAFE': {'coherence': 0.7, 'consistency': 0.3},
            'TIER_2_MODERATE': {'coherence': 0.5, 'candidate': 0.3, 'consistency': 0.2},
            'TIER_3_COMPLEX': {'coherence': 0.4, 'candidate': 0.3, 'statistical': 0.2, 'consistency': 0.1},
            'TIER_4_CRITICAL': {'coherence': 0.4, 'candidate': 0.25, 'statistical': 0.25, 'consistency': 0.1}
        }
        
        tier_weights = weights[tier]
        composite_score = 0.0
        flagged_elements = []
        
        # Calcul pondéré
        for validation_type, weight in tier_weights.items():
            if validation_type == 'coherence':
                score = validation_results.get('coherence_factuality', {}).get('score', 1.0)
            elif validation_type == 'candidate':
                score = validation_results.get('candidate', {}).get('score', 1.0)
            elif validation_type == 'statistical':
                score = validation_results.get('statistical', {}).get('score', 1.0)
            elif validation_type == 'consistency':
                score = validation_results.get('consistency', {}).get('score', 1.0)
            else:
                score = 1.0
                
            composite_score += weight * score
            
            # Collecte des éléments flagués avec sensibilité par tier
            validation_data = validation_results.get(validation_type, {})
            if validation_data.get('flagged_elements'):
                # Filtrage basé sur la sensibilité du tier
                tier_sensitivity = self._get_detection_sensitivity(tier)
                filtered_flagged = self._filter_flagged_by_sensitivity(
                    validation_data['flagged_elements'], tier_sensitivity, tier
                )
                flagged_elements.extend(filtered_flagged)
        
        # ✅ CORRECTION : Détermination du niveau de risque plus sévère
        # Prendre en compte la cohérence originale Level 1
        original_coherence = summary_data.get('coherence', 1.0)
        
        # Ajustement du score selon cohérence Level 1
        adjusted_score = composite_score
        if original_coherence < 0.3:  # 67% des cas problématiques
            adjusted_score *= 0.6  # Réduction significative
        elif original_coherence < 0.5:
            adjusted_score *= 0.8  # Réduction modérée
            
        # Détermination du risque avec seuils ajustés
        if adjusted_score < 0.4 or tier == 'TIER_4_CRITICAL' and adjusted_score < 0.6:
            risk_level = 'critical'
        elif adjusted_score < 0.6:  # ✅ Seuil plus élevé (0.5 → 0.6)
            risk_level = 'high'
        elif adjusted_score < 0.8:  # ✅ Seuil plus élevé (0.7 → 0.8)
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        # Utiliser le score ajusté comme score final
        composite_score = adjusted_score
        
        return composite_score, risk_level, flagged_elements
    
    def _extract_confidence_factors(self, validation_results: Dict, summary_data: Dict) -> Dict:
        """Extrait les facteurs de confiance pour le Niveau 3."""
        return {
            'original_grade': summary_data.get('original_grade'),
            'original_coherence': summary_data.get('coherence'),
            'original_factuality': summary_data.get('factuality'),
            'validation_tier': self.classify_summary_tier(summary_data),
            'num_validations_run': len(validation_results),
            'has_candidates': summary_data.get('fact_check_candidates_count', 0) > 0
        }
    
    def _create_fallback_result(self, summary_data: Dict, error: str) -> Level2Result:
        """Crée un résultat de fallback en cas d'erreur."""
        return Level2Result(
            summary_id=summary_data.get('id', 'unknown'),
            tier_classification='TIER_4_CRITICAL',
            coherence_factuality_score=0.0,
            candidate_validation_score=0.0,
            statistical_credibility_score=0.0,
            internal_consistency_score=0.0,
            factual_confidence=0.0,
            factual_risk_level='critical',
            processing_time_ms=0.0,
            validation_depth='error',
            validation_details={'error': error},
            flagged_elements=[f'Processing error: {error}'],
            confidence_factors={'error': True}
        )
    
    # Méthode _calculate_risk_score supprimée car non utilisée
    # La classification utilise désormais la logique directe dans classify_summary_tier()
    
    def _get_detection_sensitivity(self, tier: str) -> float:
        """
        Détermine la sensibilité de détection selon le tier.
        Plus le tier est critique, plus la détection doit être sensible.
        """
        sensitivity_by_tier = {
            'TIER_1_SAFE': 0.7,     # Moins sensible - évite les faux positifs
            'TIER_2_MODERATE': 0.5,  # Sensibilité normale
            'TIER_3_COMPLEX': 0.3,   # Plus sensible
            'TIER_4_CRITICAL': 0.1   # Très sensible - détecte tout
        }
        return sensitivity_by_tier.get(tier, 0.5)
    
    def _filter_flagged_by_sensitivity(self, flagged_elements: List[str], 
                                      sensitivity: float, tier: str) -> List[str]:
        """
        Filtre les éléments flagués selon la sensibilité du tier.
        Filtrage étalonné pour créer une vraie différence entre tiers.
        """
        if not flagged_elements:
            return []
        
        filtered = []
        for element in flagged_elements:
            should_include = self._should_include_by_tier(element, tier, sensitivity)
            if should_include:
                filtered.append(element)
        
        return filtered
    
    def _should_include_by_tier(self, element: str, tier: str, sensitivity: float) -> bool:
        """
        Détermine si un élément flagué doit être inclus selon le tier.
        """
        element_lower = element.lower()
        
        # Erreurs techniques : toujours incluses
        if 'processing error' in element_lower:
            return True
        
        # Contradictions : évaluation par tier
        if 'contradiction' in element_lower:
            if tier == 'TIER_1_SAFE':
                # TIER_1_SAFE : seulement contradictions sémantiques fortes
                semantic_indicators = ['gagné', 'perdu', 'réussi', 'échoué', 'terminé', 'inachevé']
                return any(ind in element_lower for ind in semantic_indicators)
            elif tier == 'TIER_2_MODERATE':
                # TIER_2_MODERATE : contradictions moyennes
                return len(element) > 100  # Descriptions détaillées
            else:
                # TIER_3_COMPLEX et TIER_4_CRITICAL : toutes contradictions
                return True
        
        # Incohérences d'entités : filtrage très différencié
        if 'incohérence entités' in element_lower or 'variations d\'entité' in element_lower:
            if tier == 'TIER_1_SAFE':
                # TIER_1_SAFE : quasi aucune variation d'entité (0.7 sensibilité)
                critical_indicators = ['chiffre', 'nombre', 'date', 'année', 'orthographe']
                return any(ind in element_lower for ind in critical_indicators)
            elif tier == 'TIER_2_MODERATE':
                # TIER_2_MODERATE : variations significatives seulement (0.5 sensibilité)
                # Exclure les variations grammaticales communes
                grammar_variations = [' vs ', ' de', ' le', ' la', ' ne', ' pas']
                has_grammar_only = any(var in element for var in grammar_variations)
                return not has_grammar_only or len(element) > 120
            elif tier == 'TIER_3_COMPLEX':
                # TIER_3_COMPLEX : plus tolérant (0.3 sensibilité)
                return len(element) > 80  # Éviter seulement les plus courtes
            else:
                # TIER_4_CRITICAL : tout inclure (0.1 sensibilité)
                return True
        
        # Autres types d'anomalies : graduated par tier
        if tier == 'TIER_1_SAFE':
            return len(element) > 150  # Très sélectif
        elif tier == 'TIER_2_MODERATE':
            return len(element) > 100  # Moyennement sélectif
        elif tier == 'TIER_3_COMPLEX':
            return len(element) > 50   # Peu sélectif
        else:
            return True                # Pas de filtre
# src/detection/level2_intelligent/level2_intelligent_processor.py
"""
Processeur niveau 2 intelligent avec classification différenciée.

Corrections majeures:
- Classification CRITICAL différenciée (Recoverable/Hallucination/Corrupted)
- Validation production_ready correcte
- Scores de confiance calibrés  
- Stratégies adaptatives selon type de problème
- Intégration topic overlap pour détection hallucinations
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import re
from pathlib import Path

# Import validateur pour détection hallucinations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from validation.summary_validator import SummaryValidator

logger = logging.getLogger(__name__)


class IntelligentTierClassification(Enum):
    """Classification enrichie avec sous-types CRITICAL."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"  
    MODERATE = "MODERATE"
    CRITICAL_RECOVERABLE = "CRITICAL_RECOVERABLE"          # Qualité faible mais éditable
    CRITICAL_HALLUCINATION = "CRITICAL_HALLUCINATION"      # Hallucination complète - régénération requise
    CRITICAL_CORRUPTED = "CRITICAL_CORRUPTED"              # Corruption technique - escalade manuelle


@dataclass
class IntelligentLevel2Result:
    """Résultat enrichi du niveau 2 intelligent."""
    summary_id: str
    tier_classification: IntelligentTierClassification
    is_valid: bool
    is_production_ready: bool
    validation_confidence: float
    grade_score: float
    coherence_score: float
    factuality_score: float
    issues_count: int
    level3_priority: float
    level3_strategy: str  # 'edit', 'regenerate', 'escalate'
    processing_time_ms: float
    justification: str
    diagnostic_details: Dict[str, Any]
    topic_coherence_score: Optional[float] = None
    corruption_indicators: List[str] = None
    recommended_actions: List[str] = None


class IntelligentLevel2Processor:
    """
    Processeur niveau 2 intelligent avec diagnostic précis des problèmes.
    
    Améliorations clés:
    - Détection hallucinations vs qualité médiocre
    - Classification précise par sous-type CRITICAL
    - Stratégies niveau 3 adaptées au type de problème
    - Validation production_ready correcte
    """
    
    def __init__(self, 
                 enable_hallucination_detection: bool = True,
                 enable_corruption_detection: bool = True,
                 strict_production_ready: bool = True):
        """
        Initialise le processeur intelligent.
        
        Args:
            enable_hallucination_detection: Active détection hallucinations
            enable_corruption_detection: Active détection corruptions techniques
            strict_production_ready: Mode strict pour production_ready
        """
        
        self.enable_hallucination_detection = enable_hallucination_detection
        self.enable_corruption_detection = enable_corruption_detection
        self.strict_production_ready = strict_production_ready
        
        # Initialisation validateur pour détection hallucinations
        self.validator = SummaryValidator() if enable_hallucination_detection else None
        
        # Seuils calibrés sur données réelles
        self.thresholds = {
            # Classification base
            'excellent_coherence': 0.85,
            'excellent_factuality': 0.90,
            'good_coherence': 0.70,
            'good_factuality': 0.80,
            'acceptable_coherence': 0.50,
            'acceptable_factuality': 0.70,
            
            # Détection hallucinations
            'hallucination_topic_overlap_max': 0.05,  # <5% = hallucination probable
            'hallucination_confidence_max': 0.15,      # Confiance très faible
            
            # Détection corruption
            'corruption_repetition_min': 0.30,         # >30% répétition = corruption
            'corruption_length_max': 500,              # >500 mots suspect
            'corruption_encoding_ratio_max': 0.02,     # >2% caractères corrompus
            
            # Production ready
            'production_min_coherence': 0.60,
            'production_min_factuality': 0.75,
            'production_max_issues': 3,
        }
        
        # Patterns corruption confidence_weighted
        self.corruption_patterns = [
            re.compile(r'Par\s+[\w\s]+\s+avec\s+[^\w\s]\s+le\s+[^\w\s]\s+\d+h\d+', re.IGNORECASE),
            re.compile(r'mis\s+[^\w\s]\s+jour\s+le\s+\d+\s+\w+', re.IGNORECASE),
            re.compile(r'[^\w\s]+abonner[^\w\s]+newsletter', re.IGNORECASE),
            re.compile(r'Ã[©àèêôç]'),  # Encodage corrompu
        ]

    def process_summary(self, summary_id: str, summary_data: Dict[str, Any], 
                       source_text: Optional[str] = None) -> IntelligentLevel2Result:
        """
        Traitement intelligent d'un résumé avec diagnostic précis.
        
        Args:
            summary_id: Identifiant du résumé
            summary_data: Données du résumé (métriques niveau 1, etc.)
            source_text: Texte source de l'article (pour détection hallucinations)
            
        Returns:
            IntelligentLevel2Result avec diagnostic complet
        """
        
        start_time = time.time()
        
        # Extraction données de base
        summary_text = summary_data.get('summary', '')
        strategy = summary_data.get('strategy', 'unknown')
        grade = summary_data.get('original_grade', 'D')
        coherence = float(summary_data.get('coherence', 0.0))
        factuality = float(summary_data.get('factuality', 0.0))
        issues_count = int(summary_data.get('num_issues', 0))
        
        # 1. Diagnostic approfondi
        diagnostic = self._perform_intelligent_diagnostic(
            summary_text, summary_data, source_text, strategy
        )
        
        # 2. Classification intelligente
        tier_classification = self._classify_intelligently(
            grade, coherence, factuality, issues_count, diagnostic
        )
        
        # 3. Validation production_ready corrigée
        is_production_ready = self._assess_production_readiness(
            coherence, factuality, issues_count, diagnostic, tier_classification
        )
        
        # 4. Calcul confiance calibré
        validation_confidence = self._calculate_intelligent_confidence(
            summary_data, tier_classification, diagnostic
        )
        
        # 5. Validation finale
        is_valid = self._make_validation_decision(
            validation_confidence, tier_classification, is_production_ready
        )
        
        # 6. Stratégie niveau 3 et priorisation
        level3_priority, level3_strategy, justification = self._determine_level3_strategy(
            tier_classification, diagnostic, validation_confidence
        )
        
        # 7. Recommandations actions
        recommended_actions = self._generate_action_recommendations(
            tier_classification, diagnostic, level3_strategy
        )
        
        # 8. Score grade simplifié
        grade_scores = {'A+': 1.0, 'A': 0.95, 'B+': 0.85, 'B': 0.75, 'C': 0.50, 'D': 0.30}
        grade_score = grade_scores.get(grade, 0.20)
        
        processing_time = (time.time() - start_time) * 1000.0
        
        return IntelligentLevel2Result(
            summary_id=summary_id,
            tier_classification=tier_classification,
            is_valid=is_valid,
            is_production_ready=is_production_ready,
            validation_confidence=validation_confidence,
            grade_score=grade_score,
            coherence_score=coherence,
            factuality_score=factuality,
            issues_count=issues_count,
            level3_priority=level3_priority,
            level3_strategy=level3_strategy,
            processing_time_ms=processing_time,
            justification=justification,
            diagnostic_details=diagnostic,
            topic_coherence_score=diagnostic.get('topic_overlap'),
            corruption_indicators=diagnostic.get('corruption_indicators', []),
            recommended_actions=recommended_actions
        )

    def _perform_intelligent_diagnostic(self, summary_text: str, summary_data: Dict, 
                                      source_text: Optional[str], strategy: str) -> Dict[str, Any]:
        """Diagnostic intelligent multi-dimensionnel."""
        
        diagnostic = {
            'has_hallucination': False,
            'has_corruption': False,
            'corruption_type': None,
            'corruption_indicators': [],
            'topic_overlap': None,
            'repetition_ratio': 0.0,
            'encoding_issues': 0,
            'length_category': 'normal',
            'strategy_issues': [],
            'severity_assessment': 'normal'
        }
        
        # 1. Détection hallucinations (si source disponible)
        if self.enable_hallucination_detection and source_text and self.validator:
            topic_overlap = self._calculate_topic_overlap(summary_text, source_text)
            diagnostic['topic_overlap'] = topic_overlap
            
            if topic_overlap < self.thresholds['hallucination_topic_overlap_max']:
                diagnostic['has_hallucination'] = True
                diagnostic['severity_assessment'] = 'hallucination'
        
        # 2. Détection corruption technique
        if self.enable_corruption_detection:
            corruption_analysis = self._analyze_corruption(summary_text, strategy)
            diagnostic.update(corruption_analysis)
        
        # 3. Analyse longueur et répétitions
        word_count = len(summary_text.split())
        if word_count > self.thresholds['corruption_length_max']:
            diagnostic['length_category'] = 'excessive'
            diagnostic['corruption_indicators'].append(f'length_excessive_{word_count}')
        elif word_count < 15:
            diagnostic['length_category'] = 'insufficient'
        
        # Répétitions
        repetition_ratio = self._calculate_repetition_ratio(summary_text)
        diagnostic['repetition_ratio'] = repetition_ratio
        if repetition_ratio > self.thresholds['corruption_repetition_min']:
            diagnostic['has_corruption'] = True
            diagnostic['corruption_indicators'].append(f'repetition_high_{repetition_ratio:.1%}')
        
        # 4. Analyse spécifique stratégie
        if strategy == "confidence_weighted":
            cw_issues = self._analyze_confidence_weighted_issues(summary_text)
            diagnostic['strategy_issues'] = cw_issues
            if cw_issues:
                diagnostic['has_corruption'] = True
                diagnostic['corruption_type'] = 'confidence_weighted'
        
        return diagnostic

    def _classify_intelligently(self, grade: str, coherence: float, factuality: float,
                              issues_count: int, diagnostic: Dict) -> IntelligentTierClassification:
        """Classification intelligente avec sous-types CRITICAL."""
        
        # 1. Si hallucination détectée → CRITICAL_HALLUCINATION
        if diagnostic['has_hallucination']:
            return IntelligentTierClassification.CRITICAL_HALLUCINATION
        
        # 2. Si corruption technique détectée → CRITICAL_CORRUPTED
        if diagnostic['has_corruption']:
            corruption_type = diagnostic.get('corruption_type')
            if corruption_type == 'confidence_weighted' or len(diagnostic['corruption_indicators']) >= 2:
                return IntelligentTierClassification.CRITICAL_CORRUPTED
        
        # 3. Classification normale avec sous-type CRITICAL_RECOVERABLE
        if grade in ['A+', 'A']:
            if (coherence >= self.thresholds['excellent_coherence'] and 
                factuality >= self.thresholds['excellent_factuality'] and 
                issues_count <= 1):
                return IntelligentTierClassification.EXCELLENT
            elif (coherence >= self.thresholds['good_coherence'] and 
                  factuality >= self.thresholds['good_factuality']):
                return IntelligentTierClassification.GOOD
            else:
                return IntelligentTierClassification.MODERATE
        
        elif grade in ['B+', 'B']:
            if (coherence >= self.thresholds['good_coherence'] and 
                factuality >= self.thresholds['good_factuality'] and 
                issues_count <= 2):
                return IntelligentTierClassification.GOOD
            elif (coherence >= self.thresholds['acceptable_coherence'] and 
                  factuality >= self.thresholds['acceptable_factuality']):
                return IntelligentTierClassification.MODERATE
            else:
                return IntelligentTierClassification.CRITICAL_RECOVERABLE
        
        elif grade == 'C':
            if (coherence >= self.thresholds['acceptable_coherence'] and 
                factuality >= self.thresholds['acceptable_factuality'] and
                issues_count <= 5):
                return IntelligentTierClassification.MODERATE
            else:
                return IntelligentTierClassification.CRITICAL_RECOVERABLE
        
        else:  # Grade D
            # Grade D peut être récupérable si pas trop de problèmes
            if (coherence >= 0.40 and factuality >= 0.60 and issues_count <= 8 and
                not diagnostic['has_hallucination'] and not diagnostic['has_corruption']):
                return IntelligentTierClassification.CRITICAL_RECOVERABLE
            else:
                return IntelligentTierClassification.CRITICAL_CORRUPTED

    def _assess_production_readiness(self, coherence: float, factuality: float, 
                                   issues_count: int, diagnostic: Dict,
                                   tier: IntelligentTierClassification) -> bool:
        """Évaluation production_ready corrigée."""
        
        # Jamais production ready si hallucination ou corruption sévère
        if (diagnostic['has_hallucination'] or 
            tier in [IntelligentTierClassification.CRITICAL_HALLUCINATION,
                    IntelligentTierClassification.CRITICAL_CORRUPTED]):
            return False
        
        # Seuils production en mode strict
        if self.strict_production_ready:
            return (
                coherence >= self.thresholds['production_min_coherence'] and
                factuality >= self.thresholds['production_min_factuality'] and
                issues_count <= self.thresholds['production_max_issues'] and
                tier in [IntelligentTierClassification.EXCELLENT, 
                        IntelligentTierClassification.GOOD,
                        IntelligentTierClassification.MODERATE]
            )
        else:
            # Mode standard moins strict
            return (
                coherence >= 0.50 and
                factuality >= 0.70 and
                issues_count <= 5 and
                tier != IntelligentTierClassification.CRITICAL_CORRUPTED
            )

    def _calculate_intelligent_confidence(self, summary_data: Dict, 
                                        tier: IntelligentTierClassification,
                                        diagnostic: Dict) -> float:
        """Calcul confiance intelligent calibré."""
        
        grade = summary_data.get('original_grade', 'D')
        coherence = float(summary_data.get('coherence', 0.0))
        factuality = float(summary_data.get('factuality', 0.0))
        issues_count = int(summary_data.get('num_issues', 0))
        
        # Score de base selon tier (corrigé)
        base_scores = {
            IntelligentTierClassification.EXCELLENT: 0.95,
            IntelligentTierClassification.GOOD: 0.85,
            IntelligentTierClassification.MODERATE: 0.65,
            IntelligentTierClassification.CRITICAL_RECOVERABLE: 0.35,
            IntelligentTierClassification.CRITICAL_HALLUCINATION: 0.10,
            IntelligentTierClassification.CRITICAL_CORRUPTED: 0.05
        }
        
        base_score = base_scores.get(tier, 0.20)
        
        # Ajustements selon métriques
        metric_adjustment = 0.0
        
        # Bonus qualité métriques
        if coherence > 0.80:
            metric_adjustment += 0.10
        elif coherence > 0.60:
            metric_adjustment += 0.05
        
        if factuality > 0.85:
            metric_adjustment += 0.10
        elif factuality > 0.75:
            metric_adjustment += 0.05
        
        # Malus problèmes
        if issues_count > 5:
            metric_adjustment -= 0.15
        elif issues_count > 3:
            metric_adjustment -= 0.08
        
        # Malus corruption/hallucination
        if diagnostic['has_hallucination']:
            metric_adjustment -= 0.30
        if diagnostic['has_corruption']:
            metric_adjustment -= 0.20
        
        # Score final
        final_score = max(0.0, min(1.0, base_score + metric_adjustment))
        
        return final_score

    def _make_validation_decision(self, confidence: float, 
                                tier: IntelligentTierClassification,
                                is_production_ready: bool) -> bool:
        """Décision validation finale."""
        
        # Jamais valide si hallucination ou corruption sévère
        if tier in [IntelligentTierClassification.CRITICAL_HALLUCINATION,
                   IntelligentTierClassification.CRITICAL_CORRUPTED]:
            return False
        
        # Validation basée sur confiance et production readiness
        return (
            confidence >= 0.50 and
            is_production_ready and
            tier in [IntelligentTierClassification.EXCELLENT,
                    IntelligentTierClassification.GOOD,
                    IntelligentTierClassification.MODERATE]
        )

    def _determine_level3_strategy(self, tier: IntelligentTierClassification,
                                 diagnostic: Dict, confidence: float) -> Tuple[float, str, str]:
        """Détermination stratégie niveau 3 adaptée."""
        
        # Stratégies selon type de problème
        if tier == IntelligentTierClassification.CRITICAL_HALLUCINATION:
            return 0.95, "regenerate", "Hallucination complète - régénération depuis source requise"
        
        elif tier == IntelligentTierClassification.CRITICAL_CORRUPTED:
            if diagnostic.get('corruption_type') == 'confidence_weighted':
                return 0.90, "regenerate", "Corruption confidence_weighted - régénération recommandée"
            else:
                return 0.85, "escalate", "Corruption technique - escalade manuelle requise"
        
        elif tier == IntelligentTierClassification.CRITICAL_RECOVERABLE:
            return 0.75, "edit", "Qualité récupérable - édition intelligente recommandée"
        
        elif tier == IntelligentTierClassification.MODERATE:
            if confidence < 0.60:
                return 0.40, "edit", "Amélioration qualité souhaitable"
            else:
                return 0.20, "none", "Qualité acceptable - aucune action requise"
        
        else:  # EXCELLENT, GOOD
            return 0.10, "none", "Qualité suffisante - aucune action requise"

    def _generate_action_recommendations(self, tier: IntelligentTierClassification,
                                       diagnostic: Dict, strategy: str) -> List[str]:
        """Génération recommandations actions spécifiques."""
        
        recommendations = []
        
        if strategy == "regenerate":
            recommendations.append("Régénérer résumé depuis texte source")
            if diagnostic['has_hallucination']:
                recommendations.append("Vérifier cohérence thématique avec source")
        
        elif strategy == "edit":
            if diagnostic['repetition_ratio'] > 0.20:
                recommendations.append("Supprimer répétitions excessives")
            if diagnostic['length_category'] == 'excessive':
                recommendations.append("Réduire longueur à 70-120 mots")
            if diagnostic['corruption_indicators']:
                recommendations.append("Corriger problèmes encodage")
            recommendations.append("Améliorer cohérence et factualité")
        
        elif strategy == "escalate":
            recommendations.append("Révision manuelle requise")
            recommendations.append("Vérifier intégrité données source")
        
        # Recommandations générales
        if tier in [IntelligentTierClassification.CRITICAL_HALLUCINATION,
                   IntelligentTierClassification.CRITICAL_CORRUPTED]:
            recommendations.append("Ne pas utiliser en production")
        
        return recommendations

    def _calculate_topic_overlap(self, summary: str, source_text: str) -> float:
        """Calcul overlap thématique pour détection hallucinations."""
        
        if not summary or not source_text:
            return 0.0
        
        # Extraction mots significatifs
        def extract_meaningful_words(text):
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'sur', 'pour', 'de', 'du', 'des',
                'avec', 'par', 'un', 'une', 'ce', 'cette', 'que', 'qui', 'est', 'sont'
            }
            return set(word for word in words if word not in stop_words)
        
        summary_words = extract_meaningful_words(summary)
        source_words = extract_meaningful_words(source_text)
        
        if not summary_words or not source_words:
            return 0.0
        
        intersection = len(summary_words & source_words)
        union = len(summary_words | source_words)
        
        return intersection / union if union > 0 else 0.0

    def _analyze_corruption(self, summary_text: str, strategy: str) -> Dict[str, Any]:
        """Analyse corruption technique multi-dimensionnelle."""
        
        analysis = {
            'has_corruption': False,
            'corruption_type': None,
            'corruption_indicators': []
        }
        
        # Détection patterns corruption
        for pattern in self.corruption_patterns:
            if pattern.search(summary_text):
                analysis['has_corruption'] = True
                analysis['corruption_indicators'].append(f'pattern_{pattern.pattern[:20]}')
        
        # Encodage corrompu
        encoding_issues = len(re.findall(r'Ã[©àèêôç]|â+|\\x[0-9a-fA-F]{2}', summary_text))
        if encoding_issues > len(summary_text) * 0.01:  # >1% caractères corrompus
            analysis['has_corruption'] = True
            analysis['corruption_indicators'].append(f'encoding_issues_{encoding_issues}')
        
        return analysis

    def _analyze_confidence_weighted_issues(self, summary_text: str) -> List[str]:
        """Analyse problèmes spécifiques confidence_weighted."""
        
        issues = []
        
        # Signature corruption spécifique
        if "Par Le Nouvel Obs avec" in summary_text:
            issues.append("signature_nouvel_obs_corruption")
        
        # Longueur excessive typique
        word_count = len(summary_text.split())
        if word_count > 1000:
            issues.append(f"excessive_length_{word_count}")
        
        # Répétitions phrases entières
        sentences = re.split(r'[.!?]+', summary_text)
        sentence_counts = {}
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
        
        max_repetitions = max(sentence_counts.values()) if sentence_counts else 0
        if max_repetitions >= 3:
            issues.append(f"sentence_repetition_{max_repetitions}")
        
        return issues

    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calcul ratio répétition."""
        
        words = text.split()
        if len(words) < 10:
            return 0.0
        
        from collections import Counter
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        return repeated_words / len(words)

    def process_batch(self, summaries_data: Dict, articles_data: Optional[List[Dict]] = None) -> Tuple[List[IntelligentLevel2Result], Dict[str, Any]]:
        """
        Traitement batch avec diagnostics détaillés.
        
        Args:
            summaries_data: Données résumés (format all_summaries_production.json)
            articles_data: Articles sources (optionnel)
            
        Returns:
            Tuple (résultats, statistiques)
        """
        
        # Index articles par ID
        articles_by_id = {}
        if articles_data:
            articles_by_id = {str(article['id']): article for article in articles_data}
        
        results = []
        
        for article_id, article_data in summaries_data.items():
            if 'strategies' not in article_data:
                continue
            
            # Récupération texte source
            source_text = ""
            if article_id in articles_by_id:
                source_text = articles_by_id[article_id].get('text', '')
            
            for strategy, strategy_data in article_data['strategies'].items():
                summary_id = f"{article_id}_{strategy}"
                
                # Données enrichies pour traitement
                enriched_data = strategy_data.copy()
                enriched_data.update({
                    'strategy': strategy,
                    'article_id': article_id,
                    'summary_id': summary_id
                })
                
                # Traitement intelligent
                result = self.process_summary(summary_id, enriched_data, source_text)
                results.append(result)
        
        # Calcul statistiques
        statistics = self._calculate_batch_statistics(results)
        
        return results, statistics

    def _calculate_batch_statistics(self, results: List[IntelligentLevel2Result]) -> Dict[str, Any]:
        """Calcul statistiques batch."""
        
        if not results:
            return {}
        
        total = len(results)
        
        # Distribution par tier
        tier_counts = {}
        for result in results:
            tier = result.tier_classification.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Distribution par stratégie niveau 3
        strategy_counts = {}
        for result in results:
            strategy = result.level3_strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Métriques qualité
        valid_count = sum(1 for r in results if r.is_valid)
        production_ready_count = sum(1 for r in results if r.is_production_ready)
        avg_confidence = sum(r.validation_confidence for r in results) / total
        avg_processing_time = sum(r.processing_time_ms for r in results) / total
        
        # Problèmes détectés
        hallucination_count = sum(1 for r in results 
                                if r.tier_classification == IntelligentTierClassification.CRITICAL_HALLUCINATION)
        corruption_count = sum(1 for r in results 
                             if r.tier_classification == IntelligentTierClassification.CRITICAL_CORRUPTED)
        
        return {
            'summary': {
                'total_processed': total,
                'valid': valid_count,
                'production_ready': production_ready_count,
                'validation_rate': valid_count / total * 100,
                'production_rate': production_ready_count / total * 100,
                'avg_confidence': avg_confidence,
                'avg_processing_time_ms': avg_processing_time
            },
            'tier_distribution': tier_counts,
            'level3_strategy_distribution': strategy_counts,
            'problems_detected': {
                'hallucinations': hallucination_count,
                'corruptions': corruption_count,
                'recoverable_critical': tier_counts.get('CRITICAL_RECOVERABLE', 0)
            },
            'recommended_actions': {
                'need_regeneration': strategy_counts.get('regenerate', 0),
                'need_editing': strategy_counts.get('edit', 0),
                'need_escalation': strategy_counts.get('escalate', 0),
                'ready_to_use': strategy_counts.get('none', 0)
            }
        }


# Fonction utilitaire pour migration
def create_intelligent_processor() -> IntelligentLevel2Processor:
    """Crée un processeur intelligent avec configuration optimale."""
    
    return IntelligentLevel2Processor(
        enable_hallucination_detection=True,
        enable_corruption_detection=True,
        strict_production_ready=True
    )
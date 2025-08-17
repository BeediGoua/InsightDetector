# src/detection/level3_adaptive/level3_adaptive_processor.py
"""
Processeur niveau 3 adaptatif avec stratégies différenciées.

Révolution par rapport au niveau 3 original:
- Stratégies adaptées au TYPE de problème (recoverable/hallucination/corruption)
- Seuils d'acceptation adaptatifs selon mode
- Régénération depuis source pour hallucinations
- Édition intelligente pour problèmes récupérables
- Escalade pour corruptions techniques
- Topic overlap adaptatif (plus de piège mortel)
"""

import re
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

# Imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from validation.summary_validator import SummaryValidator
from detection.level2_intelligent.level2_intelligent_processor import IntelligentTierClassification

logger = logging.getLogger(__name__)


class AdaptiveStrategy(Enum):
    """Stratégies adaptatives niveau 3."""
    EDIT_INTELLIGENT = "edit_intelligent"           # Édition intelligente pour récupérables
    REGENERATE_FROM_SOURCE = "regenerate_from_source"  # Régénération pour hallucinations
    ESCALATE_MANUAL = "escalate_manual"             # Escalade pour corruptions
    BYPASS_ACCEPTABLE = "bypass_acceptable"        # Bypass pour qualité acceptable


@dataclass
class AdaptiveLevel3Result:
    """Résultat adaptatif niveau 3."""
    summary_id: str
    strategy_applied: AdaptiveStrategy
    original_summary: str
    improved_summary: str
    is_accepted: bool
    improvement_score: float
    acceptance_criteria: Dict[str, Any]
    processing_time_ms: float
    
    # Métriques before/after
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    
    # Diagnostic
    original_tier: str
    final_tier: str
    issues_resolved: List[str]
    remaining_issues: List[str]
    
    # Métadonnées
    source_text_available: bool
    topic_overlap_before: Optional[float]
    topic_overlap_after: Optional[float]
    corrections_applied: List[str]
    failure_reason: Optional[str]


class AdaptiveLevel3Processor:
    """
    Processeur niveau 3 adaptatif résolvant les problèmes du niveau 3 original.
    
    Révolutions clés:
    - Détection préalable du TYPE de problème
    - Stratégies spécialisées par type
    - Seuils adaptatifs (fin du piège topic_overlap uniforme)
    - Régénération depuis source pour hallucinations
    - Édition intelligente pour récupérables
    - Bypass intelligent pour acceptable
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialise le processeur adaptatif.
        
        Args:
            config_path: Chemin configuration (None = défauts adaptatifs)
        """
        
        # Configuration adaptative par défaut
        self.config = self._load_adaptive_config(config_path)
        
        # Initialisations
        self.validator = SummaryValidator()
        self.cache_l2_eval = {}
        
        # Compteurs performance
        self.stats = {
            'processed': 0,
            'accepted': 0,
            'by_strategy': {},
            'by_original_tier': {},
            'processing_times': []
        }

    def _load_adaptive_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Charge configuration adaptative."""
        
        # Configuration par défaut révolutionnaire
        default_config = {
            # Seuils adaptatifs par stratégie (RÉVOLUTION vs original)
            'acceptance_criteria': {
                'edit_intelligent': {
                    'topic_overlap_after_text_min': 0.03,      # 3% vs 12% original
                    'topic_overlap_after_before_min': 0.40,    # Cohérence interne importante
                    'require_monotonic_improvement': False,     # Pas d'amélioration forcée
                    'min_final_tier': ['GOOD', 'EXCELLENT', 'MODERATE_GUARDED'],
                    'moderate_guard': {
                        'factuality_min': 0.75,  # vs 0.90 original
                        'coherence_min': 0.65,   # vs 0.80 original
                        'issues_max': 3           # vs 2 original
                    }
                },
                'regenerate_from_source': {
                    'topic_overlap_after_text_min': 0.10,      # Plus strict car régénéré
                    'topic_overlap_after_before_min': 0.0,     # Pas de contrainte vs avant
                    'require_monotonic_improvement': False,     # Nouveau contenu
                    'min_final_tier': ['GOOD', 'EXCELLENT', 'MODERATE'],
                    'factuality_absolute_min': 0.70,           # Minimum absolu
                    'coherence_absolute_min': 0.60
                },
                'bypass_acceptable': {
                    'no_modification_required': True,          # Pas de modification
                    'use_original_metrics': True
                },
                'escalate_manual': {
                    'automatic_rejection': True,               # Rejet automatique
                    'require_human_review': True
                }
            },
            
            # Paramètres génération
            'generation': {
                'edit_max_attempts': 3,
                'regenerate_max_attempts': 2,
                'target_word_range': (70, 120),
                'preserve_key_entities': True,
                'enhance_coherence': True
            },
            
            # Limites performance
            'performance': {
                'max_processing_time_ms': 5000,
                'enable_caching': True,
                'fallback_to_original_on_timeout': True
            }
        }
        
        if config_path and config_path.exists():
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = yaml.safe_load(f)
                # Merge configs (custom override default)
                default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Erreur chargement config {config_path}: {e}")
        
        return default_config

    def process_summary(self, summary_id: str, tier_classification: str,
                       original_summary: str, source_text: str = "",
                       metadata: Optional[Dict] = None) -> AdaptiveLevel3Result:
        """
        Traitement adaptatif basé sur le tier de classification niveau 2.
        
        Args:
            summary_id: Identifiant résumé
            tier_classification: Classification niveau 2 intelligent
            original_summary: Texte résumé original
            source_text: Texte source article (pour régénération)
            metadata: Métadonnées additionnelles
            
        Returns:
            AdaptiveLevel3Result avec stratégie adaptée appliquée
        """
        
        start_time = time.time()
        
        if metadata is None:
            metadata = {}
        
        # 1. Détermination stratégie adaptative
        strategy = self._determine_adaptive_strategy(tier_classification, source_text, metadata)
        
        # 2. Métriques initiales
        metrics_before = self._evaluate_summary_metrics(original_summary, source_text)
        topic_overlap_before = metrics_before.get('topic_overlap_with_source')
        
        # 3. Application stratégie spécialisée
        if strategy == AdaptiveStrategy.BYPASS_ACCEPTABLE:
            result = self._apply_bypass_strategy(summary_id, original_summary, metrics_before)
        
        elif strategy == AdaptiveStrategy.EDIT_INTELLIGENT:
            result = self._apply_edit_intelligent_strategy(
                summary_id, original_summary, source_text, metrics_before, metadata
            )
        
        elif strategy == AdaptiveStrategy.REGENERATE_FROM_SOURCE:
            result = self._apply_regenerate_strategy(
                summary_id, original_summary, source_text, metrics_before, metadata
            )
        
        elif strategy == AdaptiveStrategy.ESCALATE_MANUAL:
            result = self._apply_escalate_strategy(summary_id, original_summary, metrics_before)
        
        else:
            # Fallback sur édition
            result = self._apply_edit_intelligent_strategy(
                summary_id, original_summary, source_text, metrics_before, metadata
            )
        
        # 4. Finalisation
        processing_time = (time.time() - start_time) * 1000
        result.processing_time_ms = processing_time
        result.original_tier = tier_classification
        result.topic_overlap_before = topic_overlap_before
        
        # 5. Mise à jour statistiques
        self._update_stats(result)
        
        return result

    def _determine_adaptive_strategy(self, tier_classification: str, source_text: str,
                                   metadata: Dict) -> AdaptiveStrategy:
        """Détermination stratégie adaptative selon tier intelligent."""
        
        # Stratégies selon classification niveau 2 intelligent
        if tier_classification == "CRITICAL_HALLUCINATION":
            if source_text and len(source_text) > 100:
                return AdaptiveStrategy.REGENERATE_FROM_SOURCE
            else:
                return AdaptiveStrategy.ESCALATE_MANUAL
        
        elif tier_classification == "CRITICAL_CORRUPTED":
            strategy = metadata.get('strategy', '')
            if strategy == "confidence_weighted" and source_text:
                return AdaptiveStrategy.REGENERATE_FROM_SOURCE
            else:
                return AdaptiveStrategy.ESCALATE_MANUAL
        
        elif tier_classification == "CRITICAL_RECOVERABLE":
            return AdaptiveStrategy.EDIT_INTELLIGENT
        
        elif tier_classification in ["MODERATE", "GOOD"]:
            confidence = metadata.get('validation_confidence', 0.0)
            if confidence >= 0.70:
                return AdaptiveStrategy.BYPASS_ACCEPTABLE
            else:
                return AdaptiveStrategy.EDIT_INTELLIGENT
        
        elif tier_classification == "EXCELLENT":
            return AdaptiveStrategy.BYPASS_ACCEPTABLE
        
        else:
            # Fallback pour types non reconnus
            return AdaptiveStrategy.EDIT_INTELLIGENT

    def _apply_bypass_strategy(self, summary_id: str, original_summary: str,
                             metrics_before: Dict) -> AdaptiveLevel3Result:
        """Application stratégie bypass (qualité déjà acceptable)."""
        
        return AdaptiveLevel3Result(
            summary_id=summary_id,
            strategy_applied=AdaptiveStrategy.BYPASS_ACCEPTABLE,
            original_summary=original_summary,
            improved_summary=original_summary,  # Pas de modification
            is_accepted=True,                   # Accepté directement
            improvement_score=1.0,              # Score maximal
            acceptance_criteria={'bypass': True},
            processing_time_ms=0.0,
            metrics_before=metrics_before,
            metrics_after=metrics_before,       # Identiques
            original_tier="",
            final_tier="ACCEPTED_AS_IS",
            issues_resolved=[],
            remaining_issues=[],
            source_text_available=False,
            topic_overlap_before=None,
            topic_overlap_after=None,
            corrections_applied=[],
            failure_reason=None
        )

    def _apply_edit_intelligent_strategy(self, summary_id: str, original_summary: str,
                                       source_text: str, metrics_before: Dict,
                                       metadata: Dict) -> AdaptiveLevel3Result:
        """Application stratégie édition intelligente."""
        
        attempts = 0
        max_attempts = self.config['generation']['edit_max_attempts']
        best_result = None
        best_score = -1.0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Édition adaptative
            if attempts == 1:
                # Première tentative: correction problèmes évidents
                edited_summary = self._edit_fix_obvious_issues(original_summary)
            elif attempts == 2:
                # Deuxième tentative: amélioration cohérence
                edited_summary = self._edit_enhance_coherence(original_summary, source_text)
            else:
                # Troisième tentative: compression intelligente
                edited_summary = self._edit_intelligent_compression(original_summary, source_text)
            
            # Évaluation résultat
            metrics_after = self._evaluate_summary_metrics(edited_summary, source_text)
            is_accepted, criteria, score = self._evaluate_acceptance_edit(
                metrics_before, metrics_after, source_text
            )
            
            if score > best_score:
                best_score = score
                best_result = {
                    'summary': edited_summary,
                    'metrics': metrics_after,
                    'accepted': is_accepted,
                    'criteria': criteria,
                    'attempt': attempts
                }
            
            # Arrêt si accepté
            if is_accepted:
                break
        
        # Résultat final
        if best_result:
            issues_resolved, remaining_issues = self._analyze_issue_resolution(
                original_summary, best_result['summary'], metadata
            )
            
            return AdaptiveLevel3Result(
                summary_id=summary_id,
                strategy_applied=AdaptiveStrategy.EDIT_INTELLIGENT,
                original_summary=original_summary,
                improved_summary=best_result['summary'],
                is_accepted=best_result['accepted'],
                improvement_score=best_score,
                acceptance_criteria=best_result['criteria'],
                processing_time_ms=0.0,  # Sera mis à jour
                metrics_before=metrics_before,
                metrics_after=best_result['metrics'],
                original_tier="",
                final_tier=best_result['metrics'].get('tier_assessed', 'UNKNOWN'),
                issues_resolved=issues_resolved,
                remaining_issues=remaining_issues,
                source_text_available=bool(source_text),
                topic_overlap_before=metrics_before.get('topic_overlap_with_source'),
                topic_overlap_after=best_result['metrics'].get('topic_overlap_with_source'),
                corrections_applied=[f"edit_attempt_{best_result['attempt']}"],
                failure_reason=None if best_result['accepted'] else "edit_criteria_not_met"
            )
        else:
            # Échec total
            return self._create_failure_result(
                summary_id, AdaptiveStrategy.EDIT_INTELLIGENT, original_summary,
                metrics_before, "edit_failed_all_attempts"
            )

    def _apply_regenerate_strategy(self, summary_id: str, original_summary: str,
                                 source_text: str, metrics_before: Dict,
                                 metadata: Dict) -> AdaptiveLevel3Result:
        """Application stratégie régénération depuis source."""
        
        if not source_text or len(source_text) < 100:
            return self._create_failure_result(
                summary_id, AdaptiveStrategy.REGENERATE_FROM_SOURCE, original_summary,
                metrics_before, "insufficient_source_text"
            )
        
        attempts = 0
        max_attempts = self.config['generation']['regenerate_max_attempts']
        best_result = None
        best_score = -1.0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Régénération adaptative
            if attempts == 1:
                # Première tentative: extractif conservateur
                regenerated_summary = self._regenerate_extractive_conservative(source_text)
            else:
                # Deuxième tentative: extractif optimisé
                regenerated_summary = self._regenerate_extractive_optimized(source_text)
            
            # Évaluation résultat
            metrics_after = self._evaluate_summary_metrics(regenerated_summary, source_text)
            is_accepted, criteria, score = self._evaluate_acceptance_regenerate(
                metrics_before, metrics_after, source_text
            )
            
            if score > best_score:
                best_score = score
                best_result = {
                    'summary': regenerated_summary,
                    'metrics': metrics_after,
                    'accepted': is_accepted,
                    'criteria': criteria,
                    'attempt': attempts
                }
            
            # Arrêt si accepté
            if is_accepted:
                break
        
        # Résultat final
        if best_result:
            return AdaptiveLevel3Result(
                summary_id=summary_id,
                strategy_applied=AdaptiveStrategy.REGENERATE_FROM_SOURCE,
                original_summary=original_summary,
                improved_summary=best_result['summary'],
                is_accepted=best_result['accepted'],
                improvement_score=best_score,
                acceptance_criteria=best_result['criteria'],
                processing_time_ms=0.0,
                metrics_before=metrics_before,
                metrics_after=best_result['metrics'],
                original_tier="",
                final_tier=best_result['metrics'].get('tier_assessed', 'UNKNOWN'),
                issues_resolved=["hallucination_eliminated", "regenerated_from_source"],
                remaining_issues=[],
                source_text_available=True,
                topic_overlap_before=metrics_before.get('topic_overlap_with_source'),
                topic_overlap_after=best_result['metrics'].get('topic_overlap_with_source'),
                corrections_applied=[f"regenerate_attempt_{best_result['attempt']}"],
                failure_reason=None if best_result['accepted'] else "regenerate_criteria_not_met"
            )
        else:
            return self._create_failure_result(
                summary_id, AdaptiveStrategy.REGENERATE_FROM_SOURCE, original_summary,
                metrics_before, "regenerate_failed_all_attempts"
            )

    def _apply_escalate_strategy(self, summary_id: str, original_summary: str,
                               metrics_before: Dict) -> AdaptiveLevel3Result:
        """Application stratégie escalade manuelle."""
        
        return AdaptiveLevel3Result(
            summary_id=summary_id,
            strategy_applied=AdaptiveStrategy.ESCALATE_MANUAL,
            original_summary=original_summary,
            improved_summary=original_summary,  # Pas de modification
            is_accepted=False,                  # Rejeté pour révision manuelle
            improvement_score=0.0,
            acceptance_criteria={'escalate': True, 'require_human_review': True},
            processing_time_ms=0.0,
            metrics_before=metrics_before,
            metrics_after=metrics_before,
            original_tier="",
            final_tier="ESCALATED_MANUAL_REVIEW",
            issues_resolved=[],
            remaining_issues=["requires_human_review"],
            source_text_available=False,
            topic_overlap_before=None,
            topic_overlap_after=None,
            corrections_applied=[],
            failure_reason="escalated_for_manual_review"
        )

    def _edit_fix_obvious_issues(self, summary: str) -> str:
        """Édition: correction problèmes évidents."""
        
        corrected = summary
        
        # 1. Suppression répétitions phrases
        corrected = self._remove_sentence_repetitions(corrected)
        
        # 2. Correction encodage
        corrected = self._fix_encoding_issues(corrected)
        
        # 3. Nettoyage métadonnées parasites
        corrected = self._remove_metadata_pollution(corrected)
        
        # 4. Normalisation longueur
        corrected = self._normalize_length(corrected)
        
        return corrected

    def _edit_enhance_coherence(self, summary: str, source_text: str) -> str:
        """Édition: amélioration cohérence."""
        
        # Correction problèmes évidents d'abord
        enhanced = self._edit_fix_obvious_issues(summary)
        
        # Amélioration structure
        enhanced = self._improve_sentence_structure(enhanced)
        
        # Ajustement cohérence thématique (si source disponible)
        if source_text:
            enhanced = self._align_with_source_theme(enhanced, source_text)
        
        return enhanced

    def _edit_intelligent_compression(self, summary: str, source_text: str) -> str:
        """Édition: compression intelligente."""
        
        # Sélection phrases les plus importantes
        sentences = self._extract_key_sentences(summary, source_text)
        
        # Reconstruction cohérente
        compressed = self._reconstruct_coherent_summary(sentences)
        
        # Normalisation finale
        compressed = self._normalize_length(compressed)
        
        return compressed

    def _regenerate_extractive_conservative(self, source_text: str) -> str:
        """Régénération: extractif conservateur."""
        
        # Extraction premières phrases significatives
        sentences = self._extract_source_sentences(source_text)
        
        target_words = 90  # Cible conservative
        summary_sentences = []
        word_count = 0
        
        for sentence in sentences[:8]:  # Max 8 premières phrases
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words:
                summary_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        result = '. '.join(summary_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result

    def _regenerate_extractive_optimized(self, source_text: str) -> str:
        """Régénération: extractif optimisé."""
        
        # Extraction phrases par importance
        sentences = self._extract_source_sentences(source_text)
        scored_sentences = self._score_sentence_importance(sentences, source_text)
        
        # Sélection optimale
        target_words = 100
        selected = []
        word_count = 0
        
        for sentence, score in scored_sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words and score > 0.3:
                selected.append(sentence)
                word_count += sentence_words
        
        # Réorganisation chronologique si possible
        result = '. '.join(selected)
        if result and not result.endswith('.'):
            result += '.'
        
        return result

    def _evaluate_acceptance_edit(self, metrics_before: Dict, metrics_after: Dict,
                                source_text: str) -> Tuple[bool, Dict, float]:
        """Évaluation acceptation pour stratégie édition."""
        
        criteria = self.config['acceptance_criteria']['edit_intelligent']
        
        # Critères adaptatifs (révolutionnaires vs original)
        topic_overlap = metrics_after.get('topic_overlap_with_source', 0.0)
        topic_overlap_before = metrics_before.get('topic_overlap_with_source', 0.0)
        coherence_after = metrics_after.get('coherence', 0.0)
        factuality_after = metrics_after.get('factuality', 0.0)
        issues_after = metrics_after.get('issues_count', 0)
        
        # Critère 1: Topic overlap (adaptatif, pas de piège mortel)
        topic_ok = True
        if source_text:
            min_overlap_text = criteria['topic_overlap_after_text_min']
            min_overlap_before = criteria['topic_overlap_after_before_min']
            
            topic_ok = (
                topic_overlap >= min_overlap_text or  # Seuil absolu bas
                (topic_overlap_before > 0 and 
                 topic_overlap >= topic_overlap_before * min_overlap_before)  # Cohérence relative
            )
        
        # Critère 2: Qualité finale (moins strict)
        quality_ok = False
        if coherence_after >= 0.80 and factuality_after >= 0.85 and issues_after <= 2:
            quality_ok = True  # EXCELLENT/GOOD
        elif coherence_after >= criteria['moderate_guard']['coherence_min'] and \
             factuality_after >= criteria['moderate_guard']['factuality_min'] and \
             issues_after <= criteria['moderate_guard']['issues_max']:
            quality_ok = True  # MODERATE protégé
        
        # Critère 3: Pas de dégradation majeure (vs amélioration monotone forcée)
        no_major_degradation = True
        if coherence_after < metrics_before.get('coherence', 0.0) - 0.20:
            no_major_degradation = False
        if factuality_after < metrics_before.get('factuality', 0.0) - 0.15:
            no_major_degradation = False
        
        # Décision finale
        is_accepted = topic_ok and quality_ok and no_major_degradation
        
        # Score performance
        score = 0.0
        if topic_ok:
            score += 0.3
        if quality_ok:
            score += 0.5
        if no_major_degradation:
            score += 0.2
        
        acceptance_details = {
            'topic_overlap_ok': topic_ok,
            'quality_ok': quality_ok,
            'no_major_degradation': no_major_degradation,
            'topic_overlap_after': topic_overlap,
            'coherence_after': coherence_after,
            'factuality_after': factuality_after,
            'issues_after': issues_after
        }
        
        return is_accepted, acceptance_details, score

    def _evaluate_acceptance_regenerate(self, metrics_before: Dict, metrics_after: Dict,
                                      source_text: str) -> Tuple[bool, Dict, float]:
        """Évaluation acceptation pour stratégie régénération."""
        
        criteria = self.config['acceptance_criteria']['regenerate_from_source']
        
        topic_overlap = metrics_after.get('topic_overlap_with_source', 0.0)
        coherence_after = metrics_after.get('coherence', 0.0)
        factuality_after = metrics_after.get('factuality', 0.0)
        
        # Critères plus stricts car régénéré
        topic_ok = topic_overlap >= criteria['topic_overlap_after_text_min']  # 10%
        coherence_ok = coherence_after >= criteria['coherence_absolute_min']  # 60%
        factuality_ok = factuality_after >= criteria['factuality_absolute_min']  # 70%
        
        is_accepted = topic_ok and coherence_ok and factuality_ok
        
        score = 0.0
        if topic_ok:
            score += 0.4
        if coherence_ok:
            score += 0.3
        if factuality_ok:
            score += 0.3
        
        acceptance_details = {
            'topic_overlap_ok': topic_ok,
            'coherence_ok': coherence_ok,
            'factuality_ok': factuality_ok,
            'topic_overlap_after': topic_overlap,
            'coherence_after': coherence_after,
            'factuality_after': factuality_after
        }
        
        return is_accepted, acceptance_details, score

    def _evaluate_summary_metrics(self, summary: str, source_text: str = "") -> Dict[str, Any]:
        """Évaluation métriques résumé (version simplifiée pour niveau 3)."""
        
        metrics = {}
        
        # Métriques de base
        word_count = len(summary.split())
        metrics['word_count'] = word_count
        
        # Topic overlap si source disponible
        if source_text:
            overlap = self._calculate_topic_overlap(summary, source_text)
            metrics['topic_overlap_with_source'] = overlap
        
        # Heuristiques qualité (simplifiées pour performance)
        metrics['coherence'] = self._estimate_coherence_simple(summary)
        metrics['factuality'] = self._estimate_factuality_simple(summary, source_text)
        metrics['issues_count'] = self._count_obvious_issues(summary)
        
        return metrics

    def _calculate_topic_overlap(self, summary: str, source_text: str) -> float:
        """Calcul topic overlap."""
        
        if not summary or not source_text:
            return 0.0
        
        def extract_meaningful_words(text):
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'sur', 'pour', 'de', 'du', 'des'
            }
            return set(word for word in words if word not in stop_words)
        
        summary_words = extract_meaningful_words(summary)
        source_words = extract_meaningful_words(source_text)
        
        if not summary_words or not source_words:
            return 0.0
        
        intersection = len(summary_words & source_words)
        union = len(summary_words | source_words)
        
        return intersection / union if union > 0 else 0.0

    def _estimate_coherence_simple(self, summary: str) -> float:
        """Estimation cohérence simplifiée."""
        
        if not summary:
            return 0.0
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]
        if not sentences:
            return 0.0
        
        # Heuristiques simples
        score = 0.8  # Base optimiste
        
        # Pénalité longueur phrases
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length > 30:
            score -= 0.2
        elif avg_sentence_length < 5:
            score -= 0.3
        
        # Bonus diversité
        if len(set(sentences)) == len(sentences):  # Pas de répétitions
            score += 0.1
        
        return max(0.0, min(1.0, score))

    def _estimate_factuality_simple(self, summary: str, source_text: str) -> float:
        """Estimation factualité simplifiée."""
        
        if not summary:
            return 0.0
        
        if not source_text:
            return 0.75  # Neutre si pas de source
        
        # Overlap comme proxy factualité
        overlap = self._calculate_topic_overlap(summary, source_text)
        
        # Conversion overlap → factualité
        factuality = 0.60 + 0.35 * overlap  # 60-95% selon overlap
        
        return min(0.95, factuality)

    def _count_obvious_issues(self, summary: str) -> int:
        """Comptage issues évidentes."""
        
        issues = 0
        
        # Répétitions évidentes
        if summary.count(summary[:50]) > 1:
            issues += 2
        
        # Corruption encodage
        if re.search(r'Ã[©àèêôç]|â+', summary):
            issues += 1
        
        # Longueur problématique
        word_count = len(summary.split())
        if word_count > 200 or word_count < 20:
            issues += 1
        
        return issues

    # Méthodes utilitaires pour édition/régénération (simplifiées)
    
    def _remove_sentence_repetitions(self, text: str) -> str:
        """Suppression répétitions phrases."""
        sentences = [s.strip() for s in re.split(r'([.!?])', text) if s.strip()]
        seen = set()
        result = []
        
        for sentence in sentences:
            if sentence in '.!?':
                result.append(sentence)
            else:
                sentence_clean = sentence.lower().strip()
                if sentence_clean not in seen and len(sentence_clean) > 10:
                    seen.add(sentence_clean)
                    result.append(sentence)
        
        return ' '.join(result)

    def _fix_encoding_issues(self, text: str) -> str:
        """Correction problèmes encodage."""
        corrections = {
            'Ã©': 'é', 'Ã ': 'à', 'Ã¨': 'è', 'Ã´': 'ô', 'Ãª': 'ê', 'Ã§': 'ç', 'â': ''
        }
        for corrupt, correct in corrections.items():
            text = text.replace(corrupt, correct)
        return text

    def _remove_metadata_pollution(self, text: str) -> str:
        """Suppression métadonnées parasites."""
        # Patterns simples
        text = re.sub(r'Par\s+[\w\s]+\s+avec\s+[^\w\s]\s+le\s+[^\w\s]\s+\d+h\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'mis\s+[^\w\s]\s+jour\s+le\s+\d+\s+\w+', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()

    def _normalize_length(self, text: str, target_range: Tuple[int, int] = (70, 120)) -> str:
        """Normalisation longueur."""
        words = text.split()
        min_words, max_words = target_range
        
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        elif len(words) < min_words:
            # Pas d'allongement artificiel, juste retourner tel quel
            return text
        
        return text

    def _extract_source_sentences(self, source_text: str) -> List[str]:
        """Extraction phrases source."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', source_text) if s.strip()]
        return [s for s in sentences if len(s.split()) >= 5]  # Au moins 5 mots

    def _score_sentence_importance(self, sentences: List[str], source_text: str) -> List[Tuple[str, float]]:
        """Scoring importance phrases."""
        scored = []
        
        for sentence in sentences:
            score = 0.5  # Base
            
            # Bonus début de texte
            if sentences.index(sentence) < 3:
                score += 0.3
            
            # Bonus entités/chiffres
            if re.search(r'\b\d+\b', sentence):
                score += 0.2
            if re.search(r'\b[A-Z][a-z]+\b', sentence):
                score += 0.1
            
            scored.append((sentence, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _improve_sentence_structure(self, text: str) -> str:
        """Amélioration structure phrases (basique)."""
        # Pour l'instant, juste nettoyage espaces
        return re.sub(r'\s+', ' ', text).strip()

    def _align_with_source_theme(self, summary: str, source_text: str) -> str:
        """Alignement thème source (basique)."""
        # Pour l'instant, pas de modification
        return summary

    def _extract_key_sentences(self, summary: str, source_text: str) -> List[str]:
        """Extraction phrases clés."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]
        return sentences[:3]  # Garde top 3

    def _reconstruct_coherent_summary(self, sentences: List[str]) -> str:
        """Reconstruction résumé cohérent."""
        return '. '.join(sentences) + '.' if sentences else ""

    def _analyze_issue_resolution(self, original: str, improved: str, metadata: Dict) -> Tuple[List[str], List[str]]:
        """Analyse résolution issues."""
        resolved = []
        remaining = []
        
        # Analyse simple
        if len(improved.split()) < len(original.split()) * 0.8:
            resolved.append("length_reduced")
        
        if original.count(original[:50]) > improved.count(improved[:50]):
            resolved.append("repetitions_reduced")
        
        return resolved, remaining

    def _create_failure_result(self, summary_id: str, strategy: AdaptiveStrategy,
                             original_summary: str, metrics_before: Dict,
                             failure_reason: str) -> AdaptiveLevel3Result:
        """Création résultat échec."""
        
        return AdaptiveLevel3Result(
            summary_id=summary_id,
            strategy_applied=strategy,
            original_summary=original_summary,
            improved_summary=original_summary,
            is_accepted=False,
            improvement_score=0.0,
            acceptance_criteria={},
            processing_time_ms=0.0,
            metrics_before=metrics_before,
            metrics_after=metrics_before,
            original_tier="",
            final_tier="FAILED",
            issues_resolved=[],
            remaining_issues=["processing_failed"],
            source_text_available=False,
            topic_overlap_before=None,
            topic_overlap_after=None,
            corrections_applied=[],
            failure_reason=failure_reason
        )

    def _update_stats(self, result: AdaptiveLevel3Result):
        """Mise à jour statistiques."""
        
        self.stats['processed'] += 1
        if result.is_accepted:
            self.stats['accepted'] += 1
        
        strategy = result.strategy_applied.value
        if strategy not in self.stats['by_strategy']:
            self.stats['by_strategy'][strategy] = {'total': 0, 'accepted': 0}
        
        self.stats['by_strategy'][strategy]['total'] += 1
        if result.is_accepted:
            self.stats['by_strategy'][strategy]['accepted'] += 1
        
        self.stats['processing_times'].append(result.processing_time_ms)

    def get_statistics(self) -> Dict[str, Any]:
        """Récupération statistiques."""
        
        if self.stats['processed'] == 0:
            return {}
        
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        acceptance_rate = self.stats['accepted'] / self.stats['processed'] * 100
        
        return {
            'summary': {
                'total_processed': self.stats['processed'],
                'total_accepted': self.stats['accepted'],
                'acceptance_rate': acceptance_rate,
                'avg_processing_time_ms': avg_time
            },
            'by_strategy': self.stats['by_strategy'],
            'config_used': {
                'adaptive_thresholds': True,
                'strategies_available': [s.value for s in AdaptiveStrategy]
            }
        }


# Fonction factory
def create_adaptive_processor(config_path: Optional[Path] = None) -> AdaptiveLevel3Processor:
    """Crée processeur adaptatif avec configuration optimale."""
    
    return AdaptiveLevel3Processor(config_path)
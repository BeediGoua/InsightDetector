# src/detection/level0_prefilter_enhanced.py
"""
Préfiltre de niveau 0 amélioré avec correction des problèmes identifiés.
Intègre la validation et correction automatique des résumés corrompus.
"""

import re
import time
import unicodedata
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import Counter
import logging
import numpy as np
import pandas as pd

# Import du validateur
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from validation.summary_validator import SummaryValidator

logger = logging.getLogger(__name__)


@dataclass
class EnhancedFilterResult:
    """Résultat enrichi du pré-filtrage avec correction automatique."""
    is_valid: bool
    original_summary: str
    corrected_summary: str
    corrections_applied: List[str]
    rejection_reasons: List[str]
    word_count: int
    original_word_count: int
    repetition_score: float
    corruption_score: float
    processing_time_ms: float
    validation_details: Dict[str, Any]
    severity: str  # 'valid', 'moderate', 'critical'
    can_be_used: bool  # True si utilisable après correction


class EnhancedQualityFilter:
    """
    Filtre de qualité amélioré avec correction automatique des résumés corrompus.
    
    Corrections apportées:
    - Calibrage sur données SAINES uniquement
    - Détection et correction corruption confidence_weighted  
    - Seuils ajustés selon analyse empirique
    - Intégration validateur intelligent
    """
    
    def __init__(self, 
                 min_words: Optional[int] = None,
                 max_words: Optional[int] = None,
                 enable_auto_correction: bool = True,
                 enable_smart_calibration: bool = True,
                 strict_mode: bool = False):
        """
        Initialise le filtre enhanced.
        
        Args:
            min_words: Minimum mots (None = auto-calibrage intelligent)
            max_words: Maximum mots (None = auto-calibrage intelligent) 
            enable_auto_correction: Active correction automatique
            enable_smart_calibration: Active calibrage sur données saines
            strict_mode: Mode strict (plus conservateur)
        """
        
        # Initialisation validateur
        self.validator = SummaryValidator()
        self.enable_auto_correction = enable_auto_correction
        self.enable_smart_calibration = enable_smart_calibration
        self.strict_mode = strict_mode
        
        # Seuils corrigés basés sur analyse empirique
        if strict_mode:
            self.min_words = min_words or 20
            self.max_words = max_words or 150  # Très conservateur
            self.max_repetition_ratio = 0.15   # 15% max répétitions
            self.corruption_threshold = 0.05   # 5% max corruption
        else:
            self.min_words = min_words or 15
            self.max_words = max_words or 250  # Corrigé vs 600 original
            self.max_repetition_ratio = 0.30   # 30% max répétitions  
            self.corruption_threshold = 0.10   # 10% max corruption
        
        # Patterns corruption confidence_weighted spécifiques
        self.confidence_weighted_corruption = [
            r'Par\s+[\w\s]+\s+avec\s+[^\w\s]\s+le\s+[^\w\s]\s+\d+h\d+',  # "Par Le Nouvel Obs avec é le à"
            r'mis\s+[^\w\s]\s+jour\s+le\s+\d+\s+\w+',                   # "mis à jour le XX"
            r'[^\w\s]+abonner[^\w\s]+newsletter',                       # newsletter corrompu
        ]
        
        # Patterns encodage corruption étendus
        self.encoding_corruption_patterns = [
            r'Ã©',  # é mal encodé
            r'Ã ',  # à mal encodé
            r'Ã¨',  # è mal encodé
            r'Ã´',  # ô mal encodé
            r'Ãª',  # ê mal encodé
            r'Ã§',  # ç mal encodé
            r'â',   # caractères étranges
            r'\\x[0-9a-fA-F]{2}',  # séquences hex
            r'\\u[0-9a-fA-F]{4}',  # séquences unicode
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]',  # caractères contrôle
        ]
        
        # Compilation regex pour performance
        self.cw_corruption_regex = re.compile('|'.join(self.confidence_weighted_corruption), re.IGNORECASE)
        self.encoding_corruption_regex = re.compile('|'.join(self.encoding_corruption_patterns))
        
        # Patterns répétitions (phrases complètes)
        self.repetition_detection = re.compile(r'(.{20,}?)\1{2,}', re.IGNORECASE)

    def filter_summary(self, summary: str, metadata: Optional[Dict] = None) -> EnhancedFilterResult:
        """
        Filtre un résumé avec correction automatique si nécessaire.
        
        Args:
            summary: Texte du résumé
            metadata: Métadonnées optionnelles (strategy, article_id, etc.)
            
        Returns:
            EnhancedFilterResult avec diagnostic complet
        """
        
        start_time = time.time()
        original_summary = summary
        
        # Métadonnées par défaut
        if metadata is None:
            metadata = {}
        
        strategy = metadata.get('strategy', 'unknown')
        source_text = metadata.get('source_text', '')
        
        # 1. Validation initiale avec le validateur intelligent
        validation_result = self.validator.validate_summary(
            summary=summary,
            source_text=source_text,
            strategy=strategy
        )
        
        # 2. Correction automatique si activée et nécessaire
        corrected_summary = summary
        corrections_applied = []
        
        if (self.enable_auto_correction and 
            not validation_result['is_valid'] and 
            validation_result['can_be_corrected']):
            
            corrected_summary, corrections_applied = self.validator.correct_summary(
                summary=summary,
                source_text=source_text,
                validation_result=validation_result
            )
            
            # Re-validation après correction
            if corrections_applied:
                validation_result = self.validator.validate_summary(
                    summary=corrected_summary,
                    source_text=source_text,
                    strategy=strategy
                )
        
        # 3. Analyse spécifique préfiltre
        filter_analysis = self._analyze_for_prefilter(corrected_summary, original_summary)
        
        # 4. Décision finale
        is_valid, rejection_reasons, can_be_used = self._make_filtering_decision(
            validation_result, filter_analysis, corrections_applied
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return EnhancedFilterResult(
            is_valid=is_valid,
            original_summary=original_summary,
            corrected_summary=corrected_summary,
            corrections_applied=corrections_applied,
            rejection_reasons=rejection_reasons,
            word_count=len(corrected_summary.split()),
            original_word_count=len(original_summary.split()),
            repetition_score=validation_result.get('repetition_ratio', 0.0),
            corruption_score=filter_analysis['corruption_score'],
            processing_time_ms=processing_time,
            validation_details=validation_result,
            severity=validation_result['severity'],
            can_be_used=can_be_used
        )

    def _analyze_for_prefilter(self, summary: str, original_summary: str) -> Dict[str, Any]:
        """Analyse spécifique au préfiltre (performance, corruption, etc.)."""
        
        analysis = {
            'corruption_score': 0.0,
            'has_cw_corruption': False,
            'encoding_issues_count': 0,
            'repetition_patterns': [],
            'length_category': 'normal'
        }
        
        # Détection corruption confidence_weighted
        cw_matches = self.cw_corruption_regex.findall(summary)
        if cw_matches:
            analysis['has_cw_corruption'] = True
            analysis['corruption_score'] += 0.5
        
        # Comptage problèmes encodage
        encoding_matches = self.encoding_corruption_regex.findall(summary)
        analysis['encoding_issues_count'] = len(encoding_matches)
        analysis['corruption_score'] += len(encoding_matches) / len(summary) * 10
        
        # Détection répétitions longues
        repetition_matches = self.repetition_detection.findall(summary)
        analysis['repetition_patterns'] = repetition_matches
        if repetition_matches:
            analysis['corruption_score'] += 0.3
        
        # Catégorie longueur
        word_count = len(summary.split())
        if word_count < self.min_words:
            analysis['length_category'] = 'too_short'
        elif word_count > self.max_words:
            analysis['length_category'] = 'too_long'
        elif word_count > self.max_words * 0.8:
            analysis['length_category'] = 'long'
        elif word_count < self.min_words * 1.5:
            analysis['length_category'] = 'short'
        
        return analysis

    def _make_filtering_decision(self, validation_result: Dict, filter_analysis: Dict, 
                               corrections_applied: List[str]) -> Tuple[bool, List[str], bool]:
        """
        Décision finale de filtrage basée sur validation + analyse préfiltre.
        
        Returns:
            Tuple (is_valid, rejection_reasons, can_be_used)
        """
        
        rejection_reasons = []
        
        # 1. Rejet définitif si corruption trop élevée
        if filter_analysis['corruption_score'] > self.corruption_threshold:
            rejection_reasons.append(f"Corruption excessive: {filter_analysis['corruption_score']:.2f}")
        
        # 2. Rejet si confidence_weighted corrompu non corrigeable
        if (filter_analysis['has_cw_corruption'] and 
            'regenerated_from_source' not in corrections_applied):
            rejection_reasons.append("Corruption confidence_weighted non corrigeable")
        
        # 3. Rejet si longueur problématique après correction
        if filter_analysis['length_category'] in ['too_short', 'too_long']:
            rejection_reasons.append(f"Longueur {filter_analysis['length_category']}")
        
        # 4. Rejet si répétitions excessives non corrigées
        if (validation_result.get('repetition_ratio', 0) > self.max_repetition_ratio and
            'removed_repetitions' not in corrections_applied):
            rejection_reasons.append("Répétitions excessives non corrigées")
        
        # 5. Rejet si hallucination complète détectée
        has_hallucination = any(
            issue['type'] == 'topic_hallucination' 
            for issue in validation_result.get('issues', [])
        )
        if has_hallucination and 'regenerated_from_source' not in corrections_applied:
            rejection_reasons.append("Hallucination complète détectée")
        
        # Décision finale
        is_valid = len(rejection_reasons) == 0
        
        # Peut être utilisé si corrigé avec succès OU si seulement problèmes mineurs
        can_be_used = (
            is_valid or  # Parfaitement valide
            (len(corrections_applied) > 0 and validation_result['severity'] != 'critical') or  # Corrigé avec succès
            (validation_result['severity'] == 'moderate' and len(rejection_reasons) <= 1)  # Problèmes mineurs
        )
        
        return is_valid, rejection_reasons, can_be_used

    def filter_batch(self, summaries: List[Dict], articles_data: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[EnhancedFilterResult]]:
        """
        Filtrage en lot avec correction automatique.
        
        Args:
            summaries: Liste résumés à filtrer
            articles_data: Articles sources pour correction (optionnel)
            
        Returns:
            Tuple (résumés_valides, tous_résultats)
        """
        
        # Index articles par ID pour lookup rapide
        articles_by_id = {}
        if articles_data:
            articles_by_id = {str(article['id']): article for article in articles_data}
        
        valid_summaries = []
        all_results = []
        
        for summary_data in summaries:
            summary_text = summary_data.get('summary', '')
            strategy = summary_data.get('strategy', 'unknown')
            article_id = summary_data.get('article_id', '')
            
            # Récupération texte source si disponible
            source_text = ''
            if article_id in articles_by_id:
                source_text = articles_by_id[article_id].get('text', '')
            
            metadata = {
                'strategy': strategy,
                'article_id': article_id,
                'source_text': source_text
            }
            
            # Filtrage avec correction
            result = self.filter_summary(summary_text, metadata)
            all_results.append(result)
            
            # Ajout aux valides si utilisable
            if result.can_be_used:
                corrected_data = summary_data.copy()
                corrected_data['summary'] = result.corrected_summary
                corrected_data['corrections_applied'] = result.corrections_applied
                corrected_data['filter_result'] = {
                    'severity': result.severity,
                    'word_count': result.word_count,
                    'processing_time_ms': result.processing_time_ms
                }
                valid_summaries.append(corrected_data)
        
        return valid_summaries, all_results

    def calibrate_on_clean_data(self, data: List[Dict]) -> Dict[str, Any]:
        """
        Calibrage intelligent sur données SAINES uniquement.
        
        Corrige le problème d'auto-calibrage sur données corrompues.
        """
        
        if not self.enable_smart_calibration:
            return {
                'min_words': self.min_words,
                'max_words': self.max_words,
                'calibration_applied': False
            }
        
        logger.info("Calibrage intelligent sur données saines...")
        
        # 1. Identification données saines
        clean_data = []
        for item in data:
            summary = item.get('summary', '')
            
            # Critères de "sanité"
            word_count = len(summary.split())
            
            # Exclusion évidente corruption
            has_corruption = (
                self.cw_corruption_regex.search(summary) or
                len(self.encoding_corruption_regex.findall(summary)) > 5 or
                word_count > 500 or  # Longueur excessive évidente
                summary.count(summary.split('.')[0] if '.' in summary else summary[:50]) > 2  # Répétitions évidentes
            )
            
            if not has_corruption and 20 <= word_count <= 200:
                clean_data.append(item)
        
        if len(clean_data) < 10:
            logger.warning(f"Pas assez de données saines pour calibrage: {len(clean_data)}")
            return {
                'min_words': self.min_words,
                'max_words': self.max_words, 
                'calibration_applied': False,
                'clean_data_count': len(clean_data)
            }
        
        # 2. Calcul seuils optimaux sur données saines
        word_counts = [len(item['summary'].split()) for item in clean_data]
        
        # Percentiles pour seuils robustes
        p5 = np.percentile(word_counts, 5)
        p95 = np.percentile(word_counts, 95)
        median = np.median(word_counts)
        
        # Seuils ajustés
        calibrated_min = max(15, int(p5 * 0.8))  # 20% en dessous P5
        calibrated_max = min(300, int(p95 * 1.2))  # 20% au dessus P95
        
        self.min_words = calibrated_min
        self.max_words = calibrated_max
        
        logger.info(f"Calibrage terminé: min={calibrated_min}, max={calibrated_max} (médiane={median:.1f}, données saines={len(clean_data)})")
        
        return {
            'min_words': calibrated_min,
            'max_words': calibrated_max,
            'calibration_applied': True,
            'clean_data_count': len(clean_data),
            'total_data_count': len(data),
            'clean_ratio': len(clean_data) / len(data),
            'word_count_stats': {
                'median': median,
                'p5': p5,
                'p95': p95,
                'min': min(word_counts),
                'max': max(word_counts)
            }
        }

    def get_statistics(self, results: List[EnhancedFilterResult]) -> Dict[str, Any]:
        """Statistiques détaillées sur les résultats de filtrage amélioré."""
        
        if not results:
            return {}
        
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        corrected = sum(1 for r in results if r.corrections_applied)
        usable = sum(1 for r in results if r.can_be_used)
        
        # Statistiques corrections
        all_corrections = []
        for r in results:
            all_corrections.extend(r.corrections_applied)
        correction_counts = Counter(all_corrections)
        
        # Statistiques sévérité
        severity_counts = Counter(r.severity for r in results)
        
        # Statistiques corruption
        avg_corruption = sum(r.corruption_score for r in results) / total
        
        # Temps traitement
        avg_processing_time = sum(r.processing_time_ms for r in results) / total
        
        return {
            'summary': {
                'total_processed': total,
                'initially_valid': valid,
                'auto_corrected': corrected,
                'finally_usable': usable,
                'rejection_rate': (total - valid) / total * 100,
                'correction_rate': corrected / total * 100,
                'usability_rate': usable / total * 100
            },
            'corrections': {
                'types_applied': dict(correction_counts.most_common()),
                'total_corrections': len(all_corrections)
            },
            'severity_distribution': dict(severity_counts),
            'quality_metrics': {
                'avg_corruption_score': avg_corruption,
                'avg_processing_time_ms': avg_processing_time,
                'avg_word_count_before': sum(r.original_word_count for r in results) / total,
                'avg_word_count_after': sum(r.word_count for r in results) / total
            },
            'thresholds_used': {
                'min_words': self.min_words,
                'max_words': self.max_words,
                'max_repetition_ratio': self.max_repetition_ratio,
                'corruption_threshold': self.corruption_threshold,
                'strict_mode': self.strict_mode
            }
        }


# Fonction utilitaire pour migration facile
def create_enhanced_filter_from_data(summaries_data: Dict, articles_data: List[Dict], 
                                   strict_mode: bool = False) -> EnhancedQualityFilter:
    """
    Crée un filtre enhanced pré-calibré sur les données réelles.
    
    Args:
        summaries_data: Données résumés (all_summaries_production.json)
        articles_data: Données articles sources
        strict_mode: Mode strict ou standard
        
    Returns:
        EnhancedQualityFilter calibré et prêt à l'emploi
    """
    
    # Conversion données pour calibrage
    calibration_data = []
    for article_id, article_data in summaries_data.items():
        if 'strategies' not in article_data:
            continue
            
        for strategy, strategy_data in article_data['strategies'].items():
            calibration_data.append({
                'summary': strategy_data.get('summary', ''),
                'strategy': strategy,
                'article_id': article_id
            })
    
    # Création filtre avec auto-calibrage
    enhanced_filter = EnhancedQualityFilter(
        enable_auto_correction=True,
        enable_smart_calibration=True,
        strict_mode=strict_mode
    )
    
    # Calibrage intelligent
    calibration_result = enhanced_filter.calibrate_on_clean_data(calibration_data)
    
    logger.info(f"Filtre enhanced créé: {calibration_result}")
    
    return enhanced_filter
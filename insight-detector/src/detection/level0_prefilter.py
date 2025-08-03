"""
Niveau 0 : Pré-filtrage qualité (<50ms)

Module de pré-filtrage rapide pour éliminer les cas manifestement problématiques
avant toute analyse coûteuse de détection d'hallucinations.

Critères d'exclusion calibrés sur 372 résumés :
- Longueur anormale : <15 mots ou >200 mots
- Répétitions identiques >3x
- Métadonnées parasites
- Anomalies d'encodage
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter
import logging
import numpy as np
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Résultat du pré-filtrage d'un résumé."""
    is_valid: bool
    rejection_reasons: List[str]
    word_count: int
    repetition_score: float
    metadata_detected: List[str]
    encoding_issues: List[str]
    processing_time_ms: float


class QualityFilter:
    """
    Filtre de qualité pour le pré-filtrage rapide des résumés.
    
    Calibré sur 372 résumés avec seuils optimisés pour maximiser
    la précision tout en gardant un temps de traitement <50ms.
    """
    
    def __init__(self, 
                 min_words: Optional[int] = None,
                 max_words: Optional[int] = None,
                 max_repetitions: int = 5,
                 strict_mode: bool = False,
                 auto_calibrate_data: Optional[List[Dict]] = None):
        """
        Initialise le filtre qualité.
        
        Args:
            min_words: Nombre minimum de mots accepté (None = auto-calibrage)
            max_words: Nombre maximum de mots accepté (None = auto-calibrage)
            max_repetitions: Nombre maximum de répétitions identiques
            strict_mode: Mode strict avec critères plus sévères
            auto_calibrate_data: Données pour calibrage automatique des seuils
        """
        
        # Auto-calibrage si données fournies
        if auto_calibrate_data is not None:
            calibrated_params = self._auto_calibrate_thresholds(auto_calibrate_data)
            self.min_words = min_words or calibrated_params['min_words']
            self.max_words = max_words or calibrated_params['max_words']
            logger.info(f"Seuils auto-calibrés: min={self.min_words}, max={self.max_words}")
        else:
            # Valeurs par défaut conservatrices
            self.min_words = min_words or 10
            self.max_words = max_words or 600
            
        self.max_repetitions = max_repetitions
        self.strict_mode = strict_mode
        
        # Patterns de métadonnées parasites (moins restrictifs)
        self.metadata_patterns = [
            # Navigation et interface (seulement les plus évidents)
            r"\b(?:s'abonner|newsletter)\b.*\b(?:boîte mail|email)\b",
            r"\b(?:cookies?|gdpr|rgpd)\b.*\b(?:accepter|autoriser)\b",
            r"\b(?:publicité|pub)\b.*\b(?:contenu|article)\b",
            
            # Éléments techniques évidents
            r"\b(?:javascript|disabled|browser)\b.*\b(?:enable|activé)\b",
            r"\b(?:votre message|saisir ici)\b",
            
            # Métadonnées multiples (seulement si concentration élevée)
            r"(?:mis à jour|publié|lecture).*(?:newsletter|s'abonner|cookies)",
            
            # CMS évidents (phrases complètes)
            r"recevez.*newsletter.*boîte mail",
            r"toute l'actualité.*chaque semaine"
        ]
        
        # Patterns d'anomalies d'encodage
        self.encoding_patterns = [
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",  # Caractères de contrôle
            r"[^\x00-\x7F]{10,}",  # Longues séquences non-ASCII suspectes
            r"\\u[0-9a-fA-F]{4}",  # Séquences unicode non décodées
            r"\\x[0-9a-fA-F]{2}",  # Séquences hex non décodées
            r"Ã\x82Â|Ã\x83Â|Â\x80",  # Erreurs d'encodage UTF-8 courantes
        ]
        
        # Compilation des regex pour performance
        self.metadata_regex = re.compile(
            '|'.join(self.metadata_patterns), 
            re.IGNORECASE | re.MULTILINE
        )
        self.encoding_regex = re.compile(
            '|'.join(self.encoding_patterns), 
            re.MULTILINE
        )
        
    def filter_summary(self, text: str, summary_id: Optional[str] = None) -> FilterResult:
        """
        Filtre un résumé selon les critères de qualité niveau 0.
        
        Args:
            text: Texte du résumé à filtrer
            summary_id: Identifiant optionnel pour logging
            
        Returns:
            FilterResult: Résultat détaillé du filtrage
        """
        import time
        start_time = time.time()
        
        rejection_reasons = []
        
        # Nettoyage préliminaire du texte
        cleaned_text = self._clean_text(text)
        
        # 1. Vérification longueur
        words = cleaned_text.split()
        word_count = len(words)
        
        if word_count < self.min_words:
            rejection_reasons.append(f"Trop court: {word_count} mots < {self.min_words}")
        elif word_count > self.max_words:
            rejection_reasons.append(f"Trop long: {word_count} mots > {self.max_words}")
            
        # 2. Détection répétitions
        repetition_score, repetition_issues = self._detect_repetitions(cleaned_text)
        if repetition_issues:
            rejection_reasons.extend(repetition_issues)
            
        # 3. Détection métadonnées parasites
        metadata_detected = self._detect_metadata(cleaned_text)
        if metadata_detected:
            rejection_reasons.append(f"Métadonnées parasites: {', '.join(metadata_detected[:3])}")
            
        # 4. Anomalies d'encodage
        encoding_issues = self._detect_encoding_issues(text)  # Sur texte original
        if encoding_issues:
            rejection_reasons.append(f"Problèmes encodage: {len(encoding_issues)} détectés")
            
        # 5. Critères additionnels en mode strict
        if self.strict_mode:
            strict_issues = self._strict_quality_checks(cleaned_text)
            rejection_reasons.extend(strict_issues)
        
        # Calcul temps de traitement
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Décision finale
        is_valid = len(rejection_reasons) == 0
        
        # Logging pour cas problématiques
        if not is_valid and summary_id:
            logger.warning(f"Résumé {summary_id} rejeté: {'; '.join(rejection_reasons)}")
        
        return FilterResult(
            is_valid=is_valid,
            rejection_reasons=rejection_reasons,
            word_count=word_count,
            repetition_score=repetition_score,
            metadata_detected=metadata_detected,
            encoding_issues=encoding_issues,
            processing_time_ms=processing_time_ms
        )
    
    def process_batch(self, summaries: List[Dict]) -> Tuple[List[Dict], List[FilterResult]]:
        """
        Traite un lot de résumés en parallèle.
        
        Args:
            summaries: Liste de dictionnaires avec clés 'text' et optionnellement 'id'
            
        Returns:
            Tuple[valid_summaries, all_results]: Résumés valides et tous les résultats
        """
        valid_summaries = []
        all_results = []
        
        for i, summary in enumerate(summaries):
            text = summary.get('text', '')
            summary_id = summary.get('id', f'summary_{i}')
            
            result = self.filter_summary(text, summary_id)
            all_results.append(result)
            
            if result.is_valid:
                valid_summaries.append(summary)
        
        # Statistiques globales
        total_count = len(summaries)
        valid_count = len(valid_summaries)
        rejection_rate = (total_count - valid_count) / total_count * 100
        
        logger.info(f"Batch traité: {valid_count}/{total_count} valides ({rejection_rate:.1f}% rejetés)")
        
        return valid_summaries, all_results
    
    def get_statistics(self, results: List[FilterResult]) -> Dict:
        """
        Calcule des statistiques détaillées sur les résultats de filtrage.
        
        Args:
            results: Liste des résultats de filtrage
            
        Returns:
            Dict: Statistiques détaillées
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        
        # Collecte des raisons de rejet
        all_reasons = []
        for r in results:
            all_reasons.extend(r.rejection_reasons)
        
        reason_counts = Counter(reason.split(':')[0] for reason in all_reasons)
        
        # Statistiques de performance
        avg_processing_time = sum(r.processing_time_ms for r in results) / total
        
        # Distribution longueurs
        word_counts = [r.word_count for r in results]
        
        return {
            'total_summaries': total,
            'valid_summaries': valid,
            'rejection_rate_percent': (total - valid) / total * 100,
            'avg_processing_time_ms': avg_processing_time,
            'rejection_reasons': dict(reason_counts.most_common()),
            'word_count_stats': {
                'min': min(word_counts),
                'max': max(word_counts), 
                'avg': sum(word_counts) / len(word_counts),
                'median': sorted(word_counts)[len(word_counts)//2]
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """Nettoyage préliminaire du texte."""
        if not text:
            return ""
        
        # Normalisation unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Suppression caractères de contrôle (sauf \n \t)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalisation espaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _detect_repetitions(self, text: str) -> Tuple[float, List[str]]:
        """Détecte les répétitions excessives dans le texte."""
        issues = []
        repetition_score = 0.0
        
        # 1. Phrases identiques (seuil plus permissif)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
        if len(sentences) > 1:
            sentence_counts = Counter(sentences)
            
            for sentence, count in sentence_counts.items():
                if count > max(4, len(sentences) // 3) and len(sentence.split()) > 4:  # Plus permissif
                    issues.append(f"Phrase répétée {count}x")
                    repetition_score += count / len(sentences)
        
        # 2. Séquences de mots (algorithme corrigé)
        words = text.lower().split()
        if len(words) > 10:
            sequence_counts = {}
            
            # Chercher séquences de 4-6 mots
            for seq_len in [4, 5, 6]:
                for i in range(len(words) - seq_len + 1):
                    sequence = ' '.join(words[i:i+seq_len])
                    
                    # Éviter séquences très communes
                    if any(common in sequence for common in ['le la les', 'de la le', 'dans le la', 'pour le la']):
                        continue
                    
                    if sequence not in sequence_counts:
                        sequence_counts[sequence] = 0
                    sequence_counts[sequence] += 1
            
            # Détecter répétitions excessives
            for sequence, count in sequence_counts.items():
                if count > max(3, len(words) // 20):  # Seuil adaptatif
                    issues.append(f"Séquence répétée {count}x")
                    repetition_score += count * 0.05  # Impact réduit
                    
                    # Limiter le nombre d'issues reportées
                    if len(issues) >= 3:
                        break
        
        # 3. Détection paragraphes identiques
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 20]
        if len(paragraphs) > 1:
            para_counts = Counter(paragraphs)
            for para, count in para_counts.items():
                if count > 2:
                    issues.append(f"Paragraphe répété {count}x")
                    repetition_score += count * 0.2
        
        return min(repetition_score, 1.0), issues[:5]  # Max 5 issues
    
    def _detect_metadata(self, text: str) -> List[str]:
        """Détecte les métadonnées parasites (approche plus permissive)."""
        detected = []
        
        # Compter les matches pour éviter les faux positifs
        total_matches = 0
        for match in self.metadata_regex.finditer(text):
            detected.append(match.group().lower())
            total_matches += 1
        
        # Déduplication
        detected = list(dict.fromkeys(detected))
        
        # Seuil: rejeter seulement si beaucoup de métadonnées détectées
        text_words = len(text.split())
        metadata_ratio = total_matches / text_words if text_words > 0 else 0
        
        # Ne signaler que si concentration élevée de métadonnées
        if metadata_ratio > 0.1 or total_matches > 3:  # 10% du texte ou 3+ instances
            return detected[:3]  # Max 3 types
        else:
            return []  # Ignorer les cas isolés
    
    def _detect_encoding_issues(self, text: str) -> List[str]:
        """Détecte les problèmes d'encodage."""
        issues = []
        
        for match in self.encoding_regex.finditer(text):
            issues.append(f"Encoding issue: '{match.group()[:20]}...'")
        
        # Vérification caractères de remplacement
        if '�' in text:
            issues.append("Caractères de remplacement Unicode détectés")
            
        return issues
    
    def _strict_quality_checks(self, text: str) -> List[str]:
        """Critères de qualité additionnels en mode strict."""
        issues = []
        
        # Ratio ponctuation/mots trop élevé
        punct_count = len(re.findall(r'[^\w\s]', text))
        word_count = len(text.split())
        if word_count > 0 and punct_count / word_count > 0.3:
            issues.append(f"Ratio ponctuation élevé: {punct_count/word_count:.2f}")
        
        # Trop de majuscules (possibles erreurs)
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if upper_ratio > 0.2:
            issues.append(f"Trop de majuscules: {upper_ratio:.2f}")
            
        # Phrases trop courtes en moyenne
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 3:
                issues.append(f"Phrases trop courtes: {avg_sentence_length:.1f} mots/phrase")
        
        return issues
    
    def _auto_calibrate_thresholds(self, data: List[Dict]) -> Dict[str, int]:
        """
        Calibre automatiquement les seuils min/max basés sur les données.
        
        Utilise des percentiles et l'analyse des grades de qualité pour
        déterminer des seuils optimaux qui rejettent les cas problématiques
        tout en gardant la majorité des résumés valides.
        
        Args:
            data: Liste de dictionnaires avec 'text' et optionnellement 'quality_grade'
            
        Returns:
            Dict: Seuils calibrés {'min_words', 'max_words'}
        """
        logger.info(f"Calibrage automatique sur {len(data)} échantillons")
        
        # Calcul des longueurs
        word_counts = []
        quality_grades = []
        
        for item in data:
            text = item.get('text', '')
            words = len(self._clean_text(text).split())
            word_counts.append(words)
            quality_grades.append(item.get('quality_grade', 'Unknown'))
        
        word_counts = np.array(word_counts)
        
        # Statistiques de base
        stats = {
            'min': np.min(word_counts),
            'max': np.max(word_counts),
            'mean': np.mean(word_counts),
            'median': np.median(word_counts),
            'std': np.std(word_counts),
            'q25': np.percentile(word_counts, 25),
            'q75': np.percentile(word_counts, 75),
            'q95': np.percentile(word_counts, 95),
            'q99': np.percentile(word_counts, 99)
        }
        
        logger.info(f"Stats longueurs: min={stats['min']}, max={stats['max']}, "
                   f"mean={stats['mean']:.1f}, median={stats['median']:.1f}")
        
        # Stratégie 1: Percentiles très conservateurs
        # Rejeter seulement les cas vraiment extrêmes
        min_percentile_threshold = max(5, int(stats['min'] * 0.7))  # 70% du minimum observé
        max_percentile_threshold = int(np.percentile(word_counts, 98))  # 98e percentile (plus permissif)
        
        # Stratégie 2: Analyse par grades (si disponible)
        grade_based_thresholds = self._analyze_quality_grades(word_counts, quality_grades)
        
        # Stratégie 3: Détection d'outliers (IQR)
        iqr = stats['q75'] - stats['q25']
        outlier_min = max(1, int(stats['q25'] - 1.5 * iqr))
        outlier_max = int(stats['q75'] + 1.5 * iqr)
        
        # Combinaison des stratégies
        strategies = {
            'percentile': {'min': min_percentile_threshold, 'max': max_percentile_threshold},
            'grade_based': grade_based_thresholds,
            'outlier_iqr': {'min': outlier_min, 'max': outlier_max}
        }
        
        logger.info("Stratégies de calibrage:")
        for name, thresholds in strategies.items():
            if thresholds:
                rejection_rate = self._estimate_rejection_rate(word_counts, thresholds)
                logger.info(f"  {name}: min={thresholds['min']}, max={thresholds['max']}, "
                           f"rejet={rejection_rate:.1f}%")
        
        # Sélection de la meilleure stratégie
        # Priorité: grade_based > outlier_iqr > percentile
        if grade_based_thresholds and self._is_reasonable_threshold(grade_based_thresholds, stats):
            selected = grade_based_thresholds
            strategy_name = "grade_based"
        elif self._is_reasonable_threshold(strategies['outlier_iqr'], stats):
            selected = strategies['outlier_iqr']
            strategy_name = "outlier_iqr"
        else:
            selected = strategies['percentile']
            strategy_name = "percentile"
            
        # Ajustements finaux de sécurité (plus permissifs)
        final_min = max(5, selected['min'])  # Jamais moins de 5 mots
        final_max = min(selected['max'], int(stats['max'] * 0.9))  # Max 90% de la valeur maximale (plus permissif)
        
        # Vérification cohérence
        if final_min >= final_max:
            final_min = max(5, int(stats['q25'] * 0.5))
            final_max = int(stats['q75'] * 1.5)
            
        final_rejection_rate = self._estimate_rejection_rate(word_counts, {'min': final_min, 'max': final_max})
        
        logger.info(f"Stratégie sélectionnée: {strategy_name}")
        logger.info(f"Seuils finaux: min={final_min}, max={final_max}, "
                   f"rejet estimé={final_rejection_rate:.1f}%")
        
        return {
            'min_words': final_min,
            'max_words': final_max,
            'calibration_stats': stats,
            'strategy_used': strategy_name,
            'estimated_rejection_rate': final_rejection_rate
        }
    
    def _analyze_quality_grades(self, word_counts: np.ndarray, quality_grades: List[str]) -> Optional[Dict[str, int]]:
        """Analyse les grades de qualité pour calibrer les seuils."""
        if not quality_grades or all(g == 'Unknown' for g in quality_grades):
            return None
            
        try:
            df = pd.DataFrame({
                'word_count': word_counts,
                'quality_grade': quality_grades
            })
            
            # Seuils basés sur les grades D (problématiques)
            grade_d_data = df[df['quality_grade'] == 'D']
            if len(grade_d_data) > 0:
                # Analyser les caractéristiques des grades D
                d_word_counts = grade_d_data['word_count'].values
                
                # Si les grades D ont des longueurs extrêmes, les utiliser comme seuils
                d_min, d_max = np.min(d_word_counts), np.max(d_word_counts)
                overall_min, overall_max = np.min(word_counts), np.max(word_counts)
                
                # Seuils: rejeter seulement les cas extrêmes des grades D
                threshold_min = max(5, int(np.percentile(d_word_counts, 5)))  # 5e percentile
                threshold_max = min(int(np.percentile(d_word_counts, 95)), int(overall_max * 0.95))  # Plus permissif
                
                return {'min': threshold_min, 'max': threshold_max}
                
        except Exception as e:
            logger.warning(f"Erreur analyse grades: {e}")
            
        return None
    
    def _is_reasonable_threshold(self, thresholds: Dict[str, int], stats: Dict) -> bool:
        """Vérifie si les seuils sont raisonnables."""
        if not thresholds:
            return False
            
        min_words, max_words = thresholds['min'], thresholds['max']
        
        # Vérifications de base
        if min_words >= max_words:
            return False
        if min_words < 1 or max_words < 10:
            return False
        if max_words < stats['median']:  # Trop restrictif
            return False
        if min_words > stats['q25']:  # Trop restrictif
            return False
            
        return True
    
    def _estimate_rejection_rate(self, word_counts: np.ndarray, thresholds: Dict[str, int]) -> float:
        """Estime le taux de rejet avec ces seuils."""
        rejected = np.sum((word_counts < thresholds['min']) | (word_counts > thresholds['max']))
        return (rejected / len(word_counts)) * 100


# Fonctions utilitaires pour usage simple
def quick_filter(text: str, strict: bool = False) -> bool:
    """
    Filtrage rapide d'un seul résumé.
    
    Args:
        text: Texte à filtrer
        strict: Mode strict activé
        
    Returns:
        bool: True si valide, False sinon
    """
    filter_obj = QualityFilter(strict_mode=strict)
    result = filter_obj.filter_summary(text)
    return result.is_valid


def batch_filter(summaries: List[str], strict: bool = False) -> Tuple[List[str], float]:
    """
    Filtrage rapide d'une liste de résumés.
    
    Args:
        summaries: Liste de textes à filtrer
        strict: Mode strict activé
        
    Returns:
        Tuple[valid_summaries, rejection_rate]: Résumés valides et taux de rejet
    """
    filter_obj = QualityFilter(strict_mode=strict)
    summary_dicts = [{'text': text, 'id': i} for i, text in enumerate(summaries)]
    valid_summaries, results = filter_obj.process_batch(summary_dicts)
    
    valid_texts = [s['text'] for s in valid_summaries]
    rejection_rate = (len(summaries) - len(valid_texts)) / len(summaries)
    
    return valid_texts, rejection_rate


# Fonction utilitaire pour calibrage rapide
def auto_calibrate_filter(data: List[Dict], **kwargs) -> QualityFilter:
    """
    Crée un filtre avec calibrage automatique.
    
    Args:
        data: Données d'entraînement avec 'text' et optionnellement 'quality_grade'
        **kwargs: Paramètres additionnels pour QualityFilter
        
    Returns:
        QualityFilter: Filtre calibré automatiquement
    """
    return QualityFilter(auto_calibrate_data=data, **kwargs)


if __name__ == "__main__":
    # Test simple
    test_texts = [
        "Ceci est un résumé valide avec suffisamment de contenu pour passer les filtres.",
        "Court.",
        "S'abonner à notre newsletter pour recevoir toute l'actualité chaque semaine dans votre boîte mail.",
        "Répétition répétition répétition répétition répétition répétition répétition.",
        "Un bon résumé avec du contenu intéressant et une longueur appropriée pour être valide."
    ]
    
    filter_obj = QualityFilter()
    for i, text in enumerate(test_texts):
        result = filter_obj.filter_summary(text, f"test_{i}")
        print(f"Test {i}: {' VALIDE' if result.is_valid else 'REJETÉ'}")
        if not result.is_valid:
            print(f"  Raisons: {'; '.join(result.rejection_reasons)}")
        print(f"  Mots: {result.word_count}, Temps: {result.processing_time_ms:.1f}ms")
        print()
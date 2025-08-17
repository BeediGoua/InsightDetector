# src/validation/summary_validator.py
"""
Module de validation et correction des résumés corrompus.
Corrige les problèmes identifiés dans la génération confidence_weighted.
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

class SummaryValidator:
    """Validateur et correcteur de résumés avec détection d'hallucinations."""
    
    def __init__(self):
        """Initialise le validateur avec patterns et seuils."""
        
        # Seuils de détection
        self.max_reasonable_length = 300  # mots
        self.min_reasonable_length = 15   # mots
        self.max_repetition_ratio = 0.3   # 30% max de répétitions
        self.min_topic_overlap = 0.05     # 5% overlap minimum avec source
        
        # Patterns de corruption d'encodage
        self.encoding_corruption_patterns = [
            r'Ã©',  # é mal encodé
            r'Ã ',  # à mal encodé  
            r'Ã¨',  # è mal encodé
            r'â',   # caractères étranges
            r'\\x[0-9a-fA-F]{2}',  # séquences hex
            r'\\u[0-9a-fA-F]{4}',  # séquences unicode
        ]
        
        # Compilation regex pour performance
        self.encoding_regex = re.compile('|'.join(self.encoding_corruption_patterns))
        
        # Patterns de métadonnées parasites
        self.metadata_patterns = [
            r'Par\s+[\w\s]+\s+avec\s+[^\w\s]\s+le\s+[^\w\s]\s+\d+h\d+',  # "Par Le Nouvel Obs avec é le à 19h32"
            r'mis\s+[^\w\s]\s+jour\s+le\s+\d+\s+\w+',  # "mis à jour le XX"
            r'[^\w\s]+abonner[^\w\s]+newsletter',      # patterns newsletter corrompus
        ]
        self.metadata_regex = re.compile('|'.join(self.metadata_patterns), re.IGNORECASE)

    def validate_summary(self, summary: str, source_text: str = "", 
                        article_title: str = "", strategy: str = "") -> Dict[str, Any]:
        """
        Validation complète d'un résumé avec diagnostic détaillé.
        
        Args:
            summary: Texte du résumé à valider
            source_text: Texte source de l'article (optionnel)
            article_title: Titre de l'article (optionnel)
            strategy: Stratégie de génération utilisée
            
        Returns:
            Dict avec statut validation et détails problèmes
        """
        
        issues = []
        severity = "valid"
        
        # 1. Validation longueur
        word_count = len(summary.split())
        if word_count > self.max_reasonable_length:
            issues.append({
                'type': 'excessive_length',
                'severity': 'critical',
                'description': f'Résumé trop long: {word_count} mots (max {self.max_reasonable_length})',
                'value': word_count
            })
            severity = "critical"
        elif word_count < self.min_reasonable_length:
            issues.append({
                'type': 'insufficient_length', 
                'severity': 'moderate',
                'description': f'Résumé trop court: {word_count} mots (min {self.min_reasonable_length})',
                'value': word_count
            })
            if severity == "valid":
                severity = "moderate"
        
        # 2. Détection répétitions
        repetition_analysis = self._analyze_repetitions(summary)
        if repetition_analysis['ratio'] > self.max_repetition_ratio:
            issues.append({
                'type': 'excessive_repetition',
                'severity': 'critical',
                'description': f'Répétitions excessives: {repetition_analysis["ratio"]:.1%}',
                'details': repetition_analysis
            })
            severity = "critical"
        
        # 3. Détection corruption encodage
        encoding_issues = self._detect_encoding_corruption(summary)
        if encoding_issues:
            issues.append({
                'type': 'encoding_corruption',
                'severity': 'moderate',
                'description': f'Corruption encodage détectée: {len(encoding_issues)} problèmes',
                'details': encoding_issues
            })
            if severity == "valid":
                severity = "moderate"
        
        # 4. Détection métadonnées parasites
        metadata_issues = self._detect_metadata_pollution(summary)
        if metadata_issues:
            issues.append({
                'type': 'metadata_pollution',
                'severity': 'moderate',
                'description': f'Métadonnées parasites détectées',
                'details': metadata_issues
            })
            if severity == "valid":
                severity = "moderate"
        
        # 5. Validation cohérence thématique (si source disponible)
        if source_text:
            topic_overlap = self._calculate_topic_overlap(summary, source_text)
            if topic_overlap < self.min_topic_overlap:
                issues.append({
                    'type': 'topic_hallucination',
                    'severity': 'critical',
                    'description': f'Hallucination probable: overlap {topic_overlap:.1%} < {self.min_topic_overlap:.1%}',
                    'value': topic_overlap
                })
                severity = "critical"
        
        # 6. Validation stratégie-spécifique
        if strategy == "confidence_weighted":
            strategy_issues = self._validate_confidence_weighted_specific(summary)
            issues.extend(strategy_issues)
            if any(issue['severity'] == 'critical' for issue in strategy_issues):
                severity = "critical"
        
        return {
            'is_valid': severity == "valid",
            'severity': severity,
            'issues': issues,
            'word_count': word_count,
            'repetition_ratio': repetition_analysis['ratio'],
            'topic_overlap': self._calculate_topic_overlap(summary, source_text) if source_text else None,
            'can_be_corrected': self._assess_correctability(issues)
        }

    def correct_summary(self, summary: str, source_text: str = "", 
                       validation_result: Dict = None) -> Tuple[str, List[str]]:
        """
        Correction automatique d'un résumé corrompu.
        
        Args:
            summary: Résumé à corriger
            source_text: Texte source (pour régénération si nécessaire)
            validation_result: Résultat de validation préalable
            
        Returns:
            Tuple (résumé_corrigé, liste_corrections_appliquées)
        """
        
        if validation_result is None:
            validation_result = self.validate_summary(summary, source_text)
        
        corrected = summary
        corrections_applied = []
        
        # 1. Correction répétitions
        if any(issue['type'] == 'excessive_repetition' for issue in validation_result['issues']):
            corrected = self._fix_repetitions(corrected)
            corrections_applied.append("removed_repetitions")
        
        # 2. Correction encodage
        if any(issue['type'] == 'encoding_corruption' for issue in validation_result['issues']):
            corrected = self._fix_encoding_corruption(corrected)
            corrections_applied.append("fixed_encoding")
        
        # 3. Suppression métadonnées parasites
        if any(issue['type'] == 'metadata_pollution' for issue in validation_result['issues']):
            corrected = self._remove_metadata_pollution(corrected)
            corrections_applied.append("removed_metadata")
        
        # 4. Correction longueur excessive
        if any(issue['type'] == 'excessive_length' for issue in validation_result['issues']):
            corrected = self._trim_to_reasonable_length(corrected)
            corrections_applied.append("trimmed_length")
        
        # 5. Si hallucination complète et source disponible, proposer régénération
        if (any(issue['type'] == 'topic_hallucination' for issue in validation_result['issues']) 
            and source_text):
            # Tentative de résumé extractif simple depuis la source
            extractive_summary = self._generate_extractive_summary(source_text)
            if extractive_summary and len(extractive_summary.split()) >= self.min_reasonable_length:
                corrected = extractive_summary
                corrections_applied.append("regenerated_from_source")
        
        return corrected, corrections_applied

    def _analyze_repetitions(self, text: str) -> Dict[str, Any]:
        """Analyse détaillée des répétitions dans le texte."""
        
        # Analyse par phrases
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_counts = Counter(sentences)
        repeated_sentences = {s: count for s, count in sentence_counts.items() if count > 1}
        
        # Analyse par segments de mots
        words = text.split()
        segment_size = 10  # segments de 10 mots
        segments = []
        for i in range(len(words) - segment_size + 1):
            segment = ' '.join(words[i:i+segment_size])
            segments.append(segment)
        
        segment_counts = Counter(segments)
        repeated_segments = {s: count for s, count in segment_counts.items() if count > 1}
        
        # Calcul ratio global de répétition
        total_chars = len(text)
        repeated_chars = sum(len(s) * (count - 1) for s, count in repeated_sentences.items())
        repetition_ratio = repeated_chars / max(total_chars, 1)
        
        return {
            'ratio': repetition_ratio,
            'repeated_sentences': repeated_sentences,
            'repeated_segments': repeated_segments,
            'total_sentences': len(sentences),
            'unique_sentences': len(set(sentences))
        }

    def _detect_encoding_corruption(self, text: str) -> List[Dict[str, str]]:
        """Détecte les problèmes d'encodage dans le texte."""
        
        issues = []
        
        # Recherche patterns corruption
        matches = self.encoding_regex.finditer(text)
        for match in matches:
            issues.append({
                'pattern': match.group(),
                'position': match.span(),
                'context': text[max(0, match.start()-20):match.end()+20]
            })
        
        # Détection caractères non-ASCII suspects
        non_ascii_chars = [c for c in text if ord(c) > 127 and c not in 'àáâäèéêëìíîïòóôöùúûüÿç']
        if len(non_ascii_chars) > 10:
            issues.append({
                'pattern': 'non_ascii_excess',
                'count': len(non_ascii_chars),
                'examples': ''.join(list(set(non_ascii_chars))[:10])
            })
        
        return issues

    def _detect_metadata_pollution(self, text: str) -> List[Dict[str, Any]]:
        """Détecte les métadonnées parasites dans le texte."""
        
        issues = []
        matches = self.metadata_regex.finditer(text)
        
        for match in matches:
            issues.append({
                'pattern': match.group(),
                'position': match.span(),
                'type': 'metadata_pattern'
            })
        
        return issues

    def _calculate_topic_overlap(self, summary: str, source_text: str) -> float:
        """Calcule l'overlap thématique entre résumé et source."""
        
        if not summary or not source_text:
            return 0.0
        
        # Extraction mots significatifs (>3 chars, pas de stop words)
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'sur', 'pour', 'de', 'du', 'des',
            'avec', 'par', 'un', 'une', 'ce', 'cette', 'que', 'qui', 'est', 'sont', 'was', 'were'
        }
        
        def extract_meaningful_words(text):
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
            return set(word for word in words if word not in stop_words)
        
        summary_words = extract_meaningful_words(summary)
        source_words = extract_meaningful_words(source_text)
        
        if not summary_words or not source_words:
            return 0.0
        
        intersection = len(summary_words & source_words)
        union = len(summary_words | source_words)
        
        return intersection / union if union > 0 else 0.0

    def _validate_confidence_weighted_specific(self, summary: str) -> List[Dict[str, Any]]:
        """Validations spécifiques à la stratégie confidence_weighted."""
        
        issues = []
        
        # Pattern spécifique observé: "Par Le Nouvel Obs avec é"
        if "Par Le Nouvel Obs avec" in summary:
            issues.append({
                'type': 'confidence_weighted_corruption',
                'severity': 'critical',
                'description': 'Pattern corruption confidence_weighted détecté',
                'pattern': 'signature_corruption'
            })
        
        # Détection longueur excessive typique de confidence_weighted
        word_count = len(summary.split())
        if word_count > 1000:
            issues.append({
                'type': 'confidence_weighted_length',
                'severity': 'critical', 
                'description': f'Longueur excessive typique confidence_weighted: {word_count} mots',
                'value': word_count
            })
        
        return issues

    def _assess_correctability(self, issues: List[Dict]) -> bool:
        """Évalue si un résumé peut être corrigé automatiquement."""
        
        critical_issues = [issue for issue in issues if issue['severity'] == 'critical']
        
        # Résumé récupérable si:
        # - Pas d'hallucination complète OU
        # - Seulement problèmes techniques (répétitions, encodage, longueur)
        
        has_hallucination = any(issue['type'] == 'topic_hallucination' for issue in critical_issues)
        has_only_technical_issues = all(
            issue['type'] in ['excessive_repetition', 'excessive_length', 'encoding_corruption']
            for issue in critical_issues
        )
        
        return not has_hallucination or has_only_technical_issues

    def _fix_repetitions(self, text: str) -> str:
        """Supprime les répétitions excessives."""
        
        # Suppression répétitions de phrases complètes
        sentences = [s.strip() for s in re.split(r'([.!?]+)', text) if s.strip()]
        seen_sentences = set()
        deduplicated = []
        
        for sentence in sentences:
            # Garder ponctuations
            if sentence in '.!?':
                deduplicated.append(sentence)
                continue
                
            # Déduplication phrases de contenu
            sentence_clean = re.sub(r'\s+', ' ', sentence.strip().lower())
            if sentence_clean and sentence_clean not in seen_sentences:
                seen_sentences.add(sentence_clean)
                deduplicated.append(sentence)
        
        return ' '.join(deduplicated)

    def _fix_encoding_corruption(self, text: str) -> str:
        """Corrige les problèmes d'encodage courants."""
        
        corrections = {
            'Ã©': 'é',
            'Ã ': 'à', 
            'Ã¨': 'è',
            'Ã´': 'ô',
            'Ãª': 'ê',
            'Ã§': 'ç',
            'â': '',  # Supprime caractères étranges
        }
        
        corrected = text
        for corrupt, correct in corrections.items():
            corrected = corrected.replace(corrupt, correct)
        
        # Supprime séquences hex et unicode non décodées
        corrected = re.sub(r'\\x[0-9a-fA-F]{2}', '', corrected)
        corrected = re.sub(r'\\u[0-9a-fA-F]{4}', '', corrected)
        
        return corrected

    def _remove_metadata_pollution(self, text: str) -> str:
        """Supprime les métadonnées parasites."""
        
        # Supprime patterns métadonnées identifiés
        cleaned = self.metadata_regex.sub('', text)
        
        # Nettoyage espaces multiples
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def _trim_to_reasonable_length(self, text: str) -> str:
        """Réduit le texte à une longueur raisonnable."""
        
        words = text.split()
        if len(words) <= self.max_reasonable_length:
            return text
        
        # Garde les premières phrases jusqu'à atteindre la limite
        sentences = re.split(r'([.!?]+)', text)
        result = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_words + sentence_words <= self.max_reasonable_length:
                result.append(sentence)
                current_words += sentence_words
            else:
                break
        
        return ''.join(result).strip()

    def _generate_extractive_summary(self, source_text: str, target_words: int = 100) -> str:
        """Génère un résumé extractif simple depuis le texte source."""
        
        if not source_text or len(source_text) < 100:
            return ""
        
        # Extraction premières phrases significatives
        sentences = [s.strip() for s in re.split(r'[.!?]+', source_text) if s.strip()]
        
        summary_sentences = []
        word_count = 0
        
        for sentence in sentences[:5]:  # Max 5 premières phrases
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= target_words:
                summary_sentences.append(sentence)
                word_count += sentence_words
            else:
                break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else ""


# Fonction utilitaire pour validation batch
def validate_summaries_batch(summaries_data: Dict, articles_data: List[Dict]) -> Dict[str, Any]:
    """
    Validation en lot de tous les résumés avec rapport détaillé.
    
    Args:
        summaries_data: Données résumés (format all_summaries_production.json)
        articles_data: Données articles sources
        
    Returns:
        Rapport de validation complet
    """
    
    validator = SummaryValidator()
    
    # Index articles par ID pour lookup rapide
    articles_by_id = {str(article['id']): article for article in articles_data}
    
    validation_results = {}
    statistics = {
        'total_summaries': 0,
        'valid_summaries': 0,
        'correctable_summaries': 0,
        'critical_issues': 0,
        'by_strategy': {},
        'by_issue_type': {}
    }
    
    for article_id, article_data in summaries_data.items():
        if 'strategies' not in article_data:
            continue
            
        # Récupération article source
        source_article = articles_by_id.get(article_id, {})
        source_text = source_article.get('text', '')
        article_title = source_article.get('title', '')
        
        validation_results[article_id] = {}
        
        for strategy, strategy_data in article_data['strategies'].items():
            summary = strategy_data.get('summary', '')
            
            # Validation
            result = validator.validate_summary(
                summary=summary,
                source_text=source_text,
                article_title=article_title,
                strategy=strategy
            )
            
            validation_results[article_id][strategy] = result
            
            # Statistiques
            statistics['total_summaries'] += 1
            if result['is_valid']:
                statistics['valid_summaries'] += 1
            if result['can_be_corrected']:
                statistics['correctable_summaries'] += 1
            if result['severity'] == 'critical':
                statistics['critical_issues'] += 1
            
            # Stats par stratégie
            if strategy not in statistics['by_strategy']:
                statistics['by_strategy'][strategy] = {'total': 0, 'valid': 0, 'critical': 0}
            statistics['by_strategy'][strategy]['total'] += 1
            if result['is_valid']:
                statistics['by_strategy'][strategy]['valid'] += 1
            if result['severity'] == 'critical':
                statistics['by_strategy'][strategy]['critical'] += 1
            
            # Stats par type d'issue
            for issue in result['issues']:
                issue_type = issue['type']
                if issue_type not in statistics['by_issue_type']:
                    statistics['by_issue_type'][issue_type] = 0
                statistics['by_issue_type'][issue_type] += 1
    
    return {
        'validation_results': validation_results,
        'statistics': statistics,
        'validator_config': {
            'max_reasonable_length': validator.max_reasonable_length,
            'min_reasonable_length': validator.min_reasonable_length,
            'max_repetition_ratio': validator.max_repetition_ratio,
            'min_topic_overlap': validator.min_topic_overlap
        }
    }
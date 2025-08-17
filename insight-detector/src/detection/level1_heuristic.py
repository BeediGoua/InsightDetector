# src/detection/level1_heuristic.py
"""
Analyseur heuristique niveau 1 amélioré avec corrections des problèmes identifiés.

Corrections apportées:
- Seuils longueur corrigés (150-200 max au lieu de 400-500)
- Détection répétitions phrases complètes 
- Validation Wikidata optionnelle et non-pénalisante
- Patterns corruption confidence_weighted
- Métriques calibrées sur données réelles
"""

import re
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter
import logging
import numpy as np
import spacy
from datetime import datetime, date
import calendar

logger = logging.getLogger(__name__)


@dataclass
class HeuristicResult:
    """Résultat enrichi de l'analyse heuristique niveau 1 (compatible + enhanced)."""
    is_suspect: bool
    confidence_score: float
    risk_level: str
    issues: List[Dict[str, Any]]
    processing_time_ms: float
    word_count: int
    entities_detected: int
    suspicious_entities: int
    fact_check_candidates: List[Dict]
    priority_score: float
    enrichment_metadata: Dict[str, Any]
    severity_breakdown: Dict[str, int]
    corrections_suggested: List[str]


class HeuristicAnalyzer:
    """
    Analyseur heuristique niveau 1 amélioré avec patterns corrigés.
    
    Corrections principales:
    - Seuils longueur réalistes pour résumés (15-200 mots)
    - Détection répétitions phrases complètes
    - Validation Wikidata non-bloquante 
    - Patterns spécifiques confidence_weighted
    - Score de priorité calibré
    """
    
    def __init__(self, 
                 enable_wikidata: bool = False,
                 wikidata_timeout: float = 2.0,
                 enable_entity_validation: bool = True,
                 strict_length_limits: bool = False):
        """
        Initialise l'analyseur enhanced.
        
        Args:
            enable_wikidata: Active validation Wikidata (optionnelle)
            wikidata_timeout: Timeout requêtes Wikidata
            enable_entity_validation: Active validation entités
            strict_length_limits: Seuils longueur stricts vs standards
        """
        
        self.enable_wikidata = enable_wikidata
        self.wikidata_timeout = wikidata_timeout
        self.enable_entity_validation = enable_entity_validation
        self.strict_length_limits = strict_length_limits
        
        # Seuils corrigés basés sur analyse des 372 résumés
        if strict_length_limits:
            self.WORD_COUNT_THRESHOLDS = {
                'very_short': 15,      # <15 mots = très court
                'short': 30,           # 15-30 mots = court
                'normal_min': 30,      # 30-120 mots = normal
                'normal_max': 120,     
                'long': 150,           # 120-150 mots = long
                'very_long': 200,      # 150-200 mots = très long
                'excessive': 250       # >200 mots = excessif
            }
        else:
            self.WORD_COUNT_THRESHOLDS = {
                'very_short': 10,
                'short': 20,
                'normal_min': 20,
                'normal_max': 150,
                'long': 200,
                'very_long': 250,
                'excessive': 300
            }
        
        # Patterns corruption confidence_weighted spécifiques
        self.confidence_weighted_patterns = [
            re.compile(r'Par\s+[\w\s]+\s+avec\s+[^\w\s]\s+le\s+[^\w\s]\s+\d+h\d+', re.IGNORECASE),
            re.compile(r'mis\s+[^\w\s]\s+jour\s+le\s+\d+\s+\w+', re.IGNORECASE),
            re.compile(r'[^\w\s]+abonner[^\w\s]+newsletter', re.IGNORECASE),
            re.compile(r'Le\s+Nouvel\s+Obs\s+avec\s+[^\w\s]', re.IGNORECASE),
        ]
        
        # Patterns répétitions améliorés (phrases complètes)
        self.sentence_repetition_pattern = re.compile(r'([.!?])\s*(.{30,}?)\1\s*\2', re.IGNORECASE)
        self.paragraph_repetition_pattern = re.compile(r'(.{100,?})\.\s*\1', re.IGNORECASE)
        
        # Patterns encodage corruption
        self.encoding_corruption_patterns = [
            re.compile(r'Ã[©àèêôç]'),  # Caractères français mal encodés
            re.compile(r'â+'),          # Caractères étranges
            re.compile(r'\\x[0-9a-fA-F]{2}'),  # Séquences hex
        ]
        
        # Cache entités pour performance
        self.entity_cache = {}
        
        # Initialisation spaCy (optionnelle)
        self.nlp = None
        if enable_entity_validation:
            try:
                # Essayer français d'abord, puis anglais
                try:
                    self.nlp = spacy.load("fr_core_news_sm")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.warning("Aucun modèle spaCy disponible, validation entités désactivée")
                        self.enable_entity_validation = False
            except Exception as e:
                logger.warning(f"Erreur initialisation spaCy: {e}")
                self.enable_entity_validation = False

    def analyze_summary(self, summary: str, metadata: Optional[Dict] = None) -> HeuristicResult:
        """
        Analyse heuristique complète d'un résumé.
        
        Args:
            summary: Texte du résumé
            metadata: Métadonnées (strategy, coherence, factuality, etc.)
            
        Returns:
            HeuristicResult avec diagnostic détaillé
        """
        
        start_time = time.time()
        
        if metadata is None:
            metadata = {}
        
        # Initialisation
        issues = []
        fact_check_candidates = []
        enrichment_metadata = {}
        corrections_suggested = []
        
        # Métriques de base
        word_count = len(summary.split())
        strategy = metadata.get('strategy', 'unknown')
        
        # 1. Analyse longueur avec seuils corrigés
        length_issues = self._analyze_length_enhanced(summary, word_count, strategy)
        issues.extend(length_issues)
        
        # 2. Détection répétitions améliorée (phrases complètes)
        repetition_issues = self._analyze_repetitions_enhanced(summary)
        issues.extend(repetition_issues)
        if repetition_issues:
            corrections_suggested.append("remove_sentence_repetitions")
        
        # 3. Détection corruption confidence_weighted
        corruption_issues = self._analyze_confidence_weighted_corruption(summary)
        issues.extend(corruption_issues)
        if corruption_issues:
            corrections_suggested.append("fix_confidence_weighted_corruption")
        
        # 4. Analyse encodage corruption
        encoding_issues = self._analyze_encoding_corruption(summary)
        issues.extend(encoding_issues)
        if encoding_issues:
            corrections_suggested.append("fix_encoding_issues")
        
        # 5. Analyse statistique générale
        statistical_issues = self._analyze_statistical_anomalies_enhanced(summary)
        issues.extend(statistical_issues)
        
        # 6. Analyse syntaxique
        syntactic_issues = self._analyze_syntactic_complexity_enhanced(summary)
        issues.extend(syntactic_issues)
        
        # 7. Analyse entités (si activée)
        entities_detected = 0
        suspicious_entities = 0
        if self.enable_entity_validation and self.nlp:
            entity_issues, entities_count, suspicious_count = self._analyze_entities_enhanced(summary)
            issues.extend(entity_issues)
            entities_detected = entities_count
            suspicious_entities = suspicious_count
        
        # 8. Validation métriques existantes (coherence, factuality)
        metric_issues = self._analyze_existing_metrics_enhanced(metadata)
        issues.extend(metric_issues)
        
        # 9. Génération candidats fact-checking
        fact_check_candidates = self._generate_fact_check_candidates_enhanced(summary, entities_detected)
        
        # 10. Calcul scores et classification
        confidence_score, risk_level, priority_score = self._calculate_enhanced_scores(
            issues, word_count, entities_detected, suspicious_entities, metadata
        )
        
        # 11. Classification sévérité
        severity_breakdown = self._classify_issue_severity(issues)
        
        # 12. Métadonnées enrichissement
        enrichment_metadata = {
            'word_count': word_count,
            'sentence_count': len([s for s in re.split(r'[.!?]+', summary) if s.strip()]),
            'strategy': strategy,
            'has_corruption': len(corruption_issues) > 0,
            'repetition_ratio': self._calculate_repetition_ratio(summary),
            'encoding_quality': 1.0 - len(encoding_issues) / max(1, word_count * 0.1),
            'entity_density': entities_detected / max(1, word_count) * 100,
            'fact_check_density': len(fact_check_candidates) / max(1, word_count) * 100
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        # Détermination suspicion finale
        is_suspect = (
            confidence_score < 0.7 or
            risk_level in ['high', 'critical'] or
            len([i for i in issues if i.get('severity') == 'critical']) > 0
        )
        
        return HeuristicResult(
            is_suspect=is_suspect,
            confidence_score=confidence_score,
            risk_level=risk_level,
            issues=issues,
            processing_time_ms=processing_time,
            word_count=word_count,
            entities_detected=entities_detected,
            suspicious_entities=suspicious_entities,
            fact_check_candidates=fact_check_candidates,
            priority_score=priority_score,
            enrichment_metadata=enrichment_metadata,
            severity_breakdown=severity_breakdown,
            corrections_suggested=corrections_suggested
        )

    def _analyze_length_enhanced(self, summary: str, word_count: int, strategy: str) -> List[Dict]:
        """Analyse longueur avec seuils corrigés selon stratégie."""
        
        issues = []
        thresholds = self.WORD_COUNT_THRESHOLDS
        
        # Ajustement seuils selon stratégie
        if strategy == "confidence_weighted":
            # Plus strict pour confidence_weighted (génère souvent trop long)
            excessive_threshold = min(thresholds['excessive'], 200)
            very_long_threshold = min(thresholds['very_long'], 150)
        else:
            excessive_threshold = thresholds['excessive']
            very_long_threshold = thresholds['very_long']
        
        if word_count < thresholds['very_short']:
            issues.append({
                'type': 'longueur_trop_courte',
                'severity': 'moderate',
                'description': f"Résumé très court: {word_count} mots (min recommandé: {thresholds['very_short']})",
                'confidence': 0.7,
                'value': word_count,
                'threshold': thresholds['very_short']
            })
        
        elif word_count > excessive_threshold:
            issues.append({
                'type': 'longueur_excessive',
                'severity': 'critical',
                'description': f"Résumé excessivement long: {word_count} mots (max: {excessive_threshold})",
                'confidence': 0.9,
                'value': word_count,
                'threshold': excessive_threshold
            })
        
        elif word_count > very_long_threshold:
            issues.append({
                'type': 'longueur_suspecte',
                'severity': 'moderate',
                'description': f"Résumé très long: {word_count} mots (recommandé: <{very_long_threshold})",
                'confidence': 0.6,
                'value': word_count,
                'threshold': very_long_threshold
            })
        
        return issues

    def _analyze_repetitions_enhanced(self, summary: str) -> List[Dict]:
        """Détection répétitions phrases complètes (problème principal confidence_weighted)."""
        
        issues = []
        
        # 1. Détection répétitions phrases exactes
        sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]
        sentence_counts = Counter(sentences)
        
        repeated_sentences = {s: count for s, count in sentence_counts.items() 
                            if count > 1 and len(s) > 20}  # Seulement phrases significatives
        
        if repeated_sentences:
            max_repetitions = max(repeated_sentences.values())
            total_repetitions = sum(count - 1 for count in repeated_sentences.values())
            
            severity = 'critical' if max_repetitions >= 5 else 'moderate' if max_repetitions >= 3 else 'minor'
            
            issues.append({
                'type': 'repetition_phrases_completes',
                'severity': severity,
                'description': f"Répétition de phrases complètes détectée: {len(repeated_sentences)} phrases répétées",
                'confidence': 0.9,
                'details': {
                    'repeated_sentences': dict(list(repeated_sentences.items())[:3]),  # Top 3
                    'max_repetitions': max_repetitions,
                    'total_excess_repetitions': total_repetitions
                },
                'value': total_repetitions
            })
        
        # 2. Détection répétitions segments longs (pattern confidence_weighted)
        segment_matches = self.paragraph_repetition_pattern.findall(summary)
        if segment_matches:
            issues.append({
                'type': 'repetition_segments_longs',
                'severity': 'critical',
                'description': f"Répétition de segments longs détectée: {len(segment_matches)} occurrences",
                'confidence': 0.85,
                'details': {'repeated_segments': segment_matches[:2]},  # Exemples
                'value': len(segment_matches)
            })
        
        # 3. Calcul ratio répétition global
        repetition_ratio = self._calculate_repetition_ratio(summary)
        if repetition_ratio > 0.3:  # >30% répétition
            issues.append({
                'type': 'taux_repetition_eleve',
                'severity': 'moderate' if repetition_ratio < 0.5 else 'critical',
                'description': f"Taux de répétition élevé: {repetition_ratio:.1%}",
                'confidence': 0.8,
                'value': repetition_ratio
            })
        
        return issues

    def _analyze_confidence_weighted_corruption(self, summary: str) -> List[Dict]:
        """Détection patterns corruption spécifiques confidence_weighted."""
        
        issues = []
        
        for i, pattern in enumerate(self.confidence_weighted_patterns):
            matches = pattern.findall(summary)
            if matches:
                issues.append({
                    'type': 'corruption_confidence_weighted',
                    'severity': 'critical',
                    'description': f"Pattern corruption confidence_weighted détecté: {pattern.pattern[:50]}...",
                    'confidence': 0.95,
                    'details': {
                        'pattern_id': i,
                        'matches': matches[:3],  # Premiers 3 matches
                        'match_count': len(matches)
                    },
                    'value': len(matches)
                })
        
        # Détection signature spécifique "Par Le Nouvel Obs avec é"
        if "Par Le Nouvel Obs avec" in summary:
            issues.append({
                'type': 'signature_corruption_nouvel_obs',
                'severity': 'critical',
                'description': "Signature corruption 'Par Le Nouvel Obs avec é' détectée",
                'confidence': 0.98,
                'value': summary.count("Par Le Nouvel Obs avec")
            })
        
        return issues

    def _analyze_encoding_corruption(self, summary: str) -> List[Dict]:
        """Analyse corruption encodage avec patterns étendus."""
        
        issues = []
        total_corruption = 0
        
        for pattern in self.encoding_corruption_patterns:
            matches = pattern.findall(summary)
            if matches:
                total_corruption += len(matches)
        
        if total_corruption > 0:
            corruption_ratio = total_corruption / len(summary)
            severity = 'critical' if corruption_ratio > 0.02 else 'moderate'
            
            issues.append({
                'type': 'corruption_encodage',
                'severity': severity,
                'description': f"Corruption encodage détectée: {total_corruption} problèmes",
                'confidence': 0.8,
                'details': {
                    'corruption_count': total_corruption,
                    'corruption_ratio': corruption_ratio,
                    'text_length': len(summary)
                },
                'value': corruption_ratio
            })
        
        return issues

    def _analyze_statistical_anomalies_enhanced(self, summary: str) -> List[Dict]:
        """Analyse statistique avec seuils calibrés."""
        
        issues = []
        
        # Analyse ponctuation (seuil ajusté)
        punct_ratio = len(re.findall(r'[.,:;!?()]', summary)) / max(1, len(summary))
        if punct_ratio > 0.12:  # Abaissé de 0.15 à 0.12
            issues.append({
                'type': 'ratio_ponctuation_eleve',
                'severity': 'minor',
                'description': f"Ratio ponctuation élevé: {punct_ratio:.2%}",
                'confidence': 0.5,
                'value': punct_ratio
            })
        
        # Diversité lexicale
        words = re.findall(r'\b\w+\b', summary.lower())
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.6:  # <60% mots uniques
                issues.append({
                    'type': 'diversite_lexicale_faible',
                    'severity': 'moderate',
                    'description': f"Diversité lexicale faible: {unique_ratio:.1%}",
                    'confidence': 0.7,
                    'value': unique_ratio
                })
        
        return issues

    def _analyze_syntactic_complexity_enhanced(self, summary: str) -> List[Dict]:
        """Analyse complexité syntaxique calibrée."""
        
        issues = []
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', summary) if s.strip()]
        if not sentences:
            return issues
        
        # Longueur moyenne phrases
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg_sentence_length > 25:  # Abaissé de 30 à 25
            issues.append({
                'type': 'phrases_tres_longues',
                'severity': 'moderate',
                'description': f"Phrases très longues: {avg_sentence_length:.1f} mots/phrase",
                'confidence': 0.6,
                'value': avg_sentence_length
            })
        
        # Absence connecteurs logiques (pour textes longs)
        if len(summary.split()) > 50:
            connecteurs = ['mais', 'cependant', 'donc', 'ainsi', 'par ailleurs', 'néanmoins',
                          'however', 'therefore', 'moreover', 'nevertheless', 'furthermore']
            has_connecteurs = any(conn in summary.lower() for conn in connecteurs)
            
            if not has_connecteurs:
                issues.append({
                    'type': 'absence_connecteurs_logiques',
                    'severity': 'minor',
                    'description': "Absence de connecteurs logiques dans un texte long",
                    'confidence': 0.4,
                    'value': 0
                })
        
        return issues

    def _analyze_entities_enhanced(self, summary: str) -> Tuple[List[Dict], int, int]:
        """Analyse entités avec validation Wikidata optionnelle."""
        
        issues = []
        entities_detected = 0
        suspicious_entities = 0
        
        if not self.nlp:
            return issues, 0, 0
        
        try:
            doc = self.nlp(summary)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            entities_detected = len(entities)
            
            # Validation Wikidata optionnelle (non-bloquante)
            if self.enable_wikidata and entities:
                for entity_text, entity_type in entities[:5]:  # Limite 5 pour performance
                    validation_result = self._validate_entity_wikidata_optional(entity_text, entity_type)
                    if validation_result and validation_result.get('suspicious', False):
                        suspicious_entities += 1
                        issues.append(validation_result)
            
            # Densité entités (analyse locale)
            word_count = len(summary.split())
            if word_count > 20:
                entity_density = entities_detected / word_count
                if entity_density < 0.02:  # <2% entités
                    issues.append({
                        'type': 'densite_entites_faible',
                        'severity': 'minor',
                        'description': f"Densité d'entités faible: {entity_density:.1%}",
                        'confidence': 0.4,
                        'value': entity_density
                    })
                elif entity_density > 0.15:  # >15% entités
                    issues.append({
                        'type': 'densite_entites_elevee',
                        'severity': 'minor',
                        'description': f"Densité d'entités élevée: {entity_density:.1%}",
                        'confidence': 0.3,
                        'value': entity_density
                    })
        
        except Exception as e:
            logger.warning(f"Erreur analyse entités: {e}")
        
        return issues, entities_detected, suspicious_entities

    def _validate_entity_wikidata_optional(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """Validation Wikidata optionnelle et non-pénalisante."""
        
        cache_key = f"{entity_text}_{entity_type}"
        
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': entity_text,
                'limit': 3,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=self.wikidata_timeout)
            if response.status_code == 200:
                results = response.json()
                found = len(results) > 1 and len(results[1]) > 0
                
                if not found and len(entity_text) > 10:  # Seulement entités significatives
                    result = {
                        'type': 'entite_non_verifiee_wikidata',
                        'severity': 'minor',  # Abaissé de moderate à minor
                        'description': f"Entité '{entity_text}' non trouvée dans Wikidata",
                        'confidence': 0.3,  # Abaissé de 0.6 à 0.3
                        'entity': entity_text,
                        'suspicious': True
                    }
                    self.entity_cache[cache_key] = result
                    return result
                
                self.entity_cache[cache_key] = None
                return None
        
        except requests.RequestException:
            # Erreur réseau = pas de pénalité
            self.entity_cache[cache_key] = None
            return None
        
        return None

    def _analyze_existing_metrics_enhanced(self, metadata: Dict) -> List[Dict]:
        """Analyse métriques existantes avec seuils calibrés."""
        
        issues = []
        
        coherence = metadata.get('coherence')
        factuality = metadata.get('factuality')
        grade = metadata.get('original_grade')
        
        if coherence is not None:
            if coherence < 0.3:  # Très faible
                issues.append({
                    'type': 'coherence_tres_faible',
                    'severity': 'critical',
                    'description': f"Cohérence très faible: {coherence:.3f}",
                    'confidence': 0.8,
                    'value': coherence
                })
            elif coherence < 0.5:  # Faible
                issues.append({
                    'type': 'coherence_faible',
                    'severity': 'moderate',
                    'description': f"Cohérence faible: {coherence:.3f}",
                    'confidence': 0.6,
                    'value': coherence
                })
        
        if factuality is not None:
            if factuality < 0.7:  # Seuil abaissé de 0.85 à 0.7
                issues.append({
                    'type': 'factualite_faible',
                    'severity': 'moderate',
                    'description': f"Factualité en dessous du seuil: {factuality:.3f}",
                    'confidence': 0.6,
                    'value': factuality
                })
        
        if grade and grade in ['D']:  # Seulement grade D = critique
            issues.append({
                'type': 'grade_problematique',
                'severity': 'critical',
                'description': f"Grade de qualité {grade} (critique)",
                'confidence': 0.9,
                'value': grade
            })
        elif grade and grade in ['C']:
            issues.append({
                'type': 'grade_mediocre',
                'severity': 'moderate',
                'description': f"Grade de qualité {grade} (médiocre)",
                'confidence': 0.7,
                'value': grade
            })
        
        # Corrélation suspecte cohérence/factualité
        if coherence is not None and factuality is not None:
            if coherence < 0.4 and factuality > 0.9:
                issues.append({
                    'type': 'correlation_suspecte',
                    'severity': 'moderate',
                    'description': f"Corrélation suspecte: cohérence faible ({coherence:.3f}) mais factualité élevée ({factuality:.3f})",
                    'confidence': 0.7,
                    'value': abs(coherence - factuality)
                })
        
        return issues

    def _generate_fact_check_candidates_enhanced(self, summary: str, entities_count: int) -> List[Dict]:
        """Génération candidats fact-checking calibrée."""
        
        candidates = []
        
        # 1. Entités nommées (si détectées)
        if entities_count > 0:
            candidates.append({
                'type': 'entity_verification',
                'priority': 0.7,
                'description': f'Vérification des {entities_count} entités détectées'
            })
        
        # 2. Dates et chiffres
        dates = re.findall(r'\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|\w+)\s+\d{4}\b', summary, re.IGNORECASE)
        numbers = re.findall(r'\b\d{1,3}(?:\s\d{3})*(?:\s(?:millions?|milliards?))?b', summary)
        
        if dates:
            candidates.append({
                'type': 'date_verification',
                'priority': 0.8,
                'description': f'Vérification des dates: {dates[:2]}'
            })
        
        if numbers:
            candidates.append({
                'type': 'number_verification', 
                'priority': 0.6,
                'description': f'Vérification des chiffres: {numbers[:3]}'
            })
        
        # 3. Affirmations factuelles (patterns simples)
        factual_patterns = [
            r'\b(?:a\s+(?:déclaré|annoncé|confirmé|révélé))\b',
            r'\b(?:selon|d\'après)\s+[\w\s]+\b',
            r'\b\d+%\s+des?\b'
        ]
        
        for pattern in factual_patterns:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            if matches:
                candidates.append({
                    'type': 'factual_claim',
                    'priority': 0.5,
                    'description': f'Vérification affirmation: {pattern}'
                })
        
        return candidates[:5]  # Limite 5 candidats

    def _calculate_enhanced_scores(self, issues: List[Dict], word_count: int, 
                                 entities_detected: int, suspicious_entities: int,
                                 metadata: Dict) -> Tuple[float, str, float]:
        """Calcul scores de confiance, risque et priorité calibrés."""
        
        # Score confiance de base (optimiste)
        base_confidence = 0.8
        
        # Pénalités par sévérité (calibrées moins pénalisantes)
        severity_penalties = {
            'critical': 0.3,
            'moderate': 0.15,
            'minor': 0.05
        }
        
        confidence_penalty = 0
        for issue in issues:
            severity = issue.get('severity', 'minor')
            confidence_penalty += severity_penalties.get(severity, 0.05)
        
        # Bonus qualité
        quality_bonus = 0
        if word_count >= 30 and word_count <= 150:  # Longueur optimale
            quality_bonus += 0.1
        if entities_detected > 0:  # Présence entités
            quality_bonus += 0.05
        
        # Score final
        confidence_score = max(0.0, min(1.0, base_confidence - confidence_penalty + quality_bonus))
        
        # Niveau de risque
        critical_issues = len([i for i in issues if i.get('severity') == 'critical'])
        if critical_issues >= 2:
            risk_level = 'critical'
        elif critical_issues >= 1:
            risk_level = 'high'
        elif len(issues) >= 5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Score priorité (pour niveau 3)
        priority_score = 0.0
        if risk_level in ['high', 'critical']:
            priority_score = 0.8 + min(0.2, critical_issues * 0.1)
        elif confidence_score < 0.5:
            priority_score = 0.6
        elif len(issues) >= 3:
            priority_score = 0.4
        
        return confidence_score, risk_level, priority_score

    def _classify_issue_severity(self, issues: List[Dict]) -> Dict[str, int]:
        """Classification sévérité des issues."""
        
        severity_counts = {'critical': 0, 'moderate': 0, 'minor': 0}
        
        for issue in issues:
            severity = issue.get('severity', 'minor')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return severity_counts

    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calcul ratio répétition global."""
        
        words = text.split()
        if len(words) < 10:
            return 0.0
        
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        return repeated_words / len(words)

    def analyze_batch(self, summaries: List[Dict], 
                     enable_progress: bool = True) -> Tuple[List[Dict], List[HeuristicResult]]:
        """
        Analyse en lot de résumés avec résultats enrichis.
        
        Args:
            summaries: Liste résumés avec métadonnées
            enable_progress: Affichage progression
            
        Returns:
            Tuple (résumés_enrichis, résultats_analyse)
        """
        
        enriched_summaries = []
        analysis_results = []
        
        total = len(summaries)
        
        for i, summary_data in enumerate(summaries):
            if enable_progress and i % 50 == 0:
                logger.info(f"Analyse niveau 1 enhanced: {i}/{total} ({i/total:.1%})")
            
            summary_text = summary_data.get('summary', '')
            metadata = {
                'strategy': summary_data.get('strategy', 'unknown'),
                'coherence': summary_data.get('coherence'),
                'factuality': summary_data.get('factuality'),
                'original_grade': summary_data.get('original_grade')
            }
            
            # Analyse enhanced
            result = self.analyze_summary(summary_text, metadata)
            analysis_results.append(result)
            
            # Enrichissement données
            enriched_data = summary_data.copy()
            enriched_data.update({
                'heuristic_result': {
                    'is_suspect': result.is_suspect,
                    'confidence_score': result.confidence_score,
                    'risk_level': result.risk_level,
                    'num_issues': len(result.issues),
                    'priority_score': result.priority_score,
                    'processing_time_ms': result.processing_time_ms
                },
                'issues_detected': [
                    {
                        'type': issue['type'],
                        'severity': issue['severity'],
                        'description': issue['description']
                    } for issue in result.issues
                ],
                'fact_check_candidates_count': len(result.fact_check_candidates),
                'corrections_suggested': result.corrections_suggested,
                'needs_fact_check': len(result.fact_check_candidates) > 0
            })
            
            enriched_summaries.append(enriched_data)
        
        if enable_progress:
            logger.info(f"Analyse niveau 1 enhanced terminée: {total} résumés traités")
        
        return enriched_summaries, analysis_results

    def get_analysis_statistics(self, results: List[HeuristicResult]) -> Dict[str, Any]:
        """Statistiques détaillées sur les analyses."""
        
        if not results:
            return {}
        
        total = len(results)
        suspects = sum(1 for r in results if r.is_suspect)
        
        # Distribution sévérité
        severity_stats = {'critical': 0, 'moderate': 0, 'minor': 0}
        issue_types = Counter()
        
        for result in results:
            for issue in result.issues:
                severity = issue.get('severity', 'minor')
                if severity in severity_stats:
                    severity_stats[severity] += 1
                issue_types[issue['type']] += 1
        
        # Statistiques performance
        avg_processing_time = sum(r.processing_time_ms for r in results) / total
        
        # Distribution scores
        confidence_scores = [r.confidence_score for r in results]
        priority_scores = [r.priority_score for r in results]
        
        return {
            'summary': {
                'total_analyzed': total,
                'suspects_detected': suspects,
                'suspect_rate': suspects / total * 100,
                'avg_processing_time_ms': avg_processing_time
            },
            'severity_distribution': severity_stats,
            'top_issue_types': dict(issue_types.most_common(10)),
            'score_statistics': {
                'confidence': {
                    'mean': np.mean(confidence_scores),
                    'median': np.median(confidence_scores),
                    'std': np.std(confidence_scores)
                },
                'priority': {
                    'mean': np.mean(priority_scores),
                    'median': np.median(priority_scores),
                    'std': np.std(priority_scores)
                }
            },
            'risk_level_distribution': Counter(r.risk_level for r in results),
            'word_count_stats': {
                'mean': np.mean([r.word_count for r in results]),
                'median': np.median([r.word_count for r in results]),
                'min': min(r.word_count for r in results),
                'max': max(r.word_count for r in results)
            }
        }


# Alias pour compatibilité ascendante
EnhancedHeuristicAnalyzer = HeuristicAnalyzer
EnhancedHeuristicResult = HeuristicResult


# Test simple si exécuté directement
if __name__ == "__main__":
    analyzer = HeuristicAnalyzer(enable_wikidata=False)
    
    test_summary = "Par Le Nouvel Obs avec é le à 14h30. Des chercheurs ont développé une nouvelle technologie. Des chercheurs ont développé une nouvelle technologie."
    
    result = analyzer.analyze_summary(test_summary, {'strategy': 'confidence_weighted'})
    
    print(f"Suspect: {result.is_suspect}")
    print(f"Confiance: {result.confidence_score:.2f}")
    print(f"Issues: {len(result.issues)}")
    print(f"Corrections suggérées: {result.corrections_suggested}")
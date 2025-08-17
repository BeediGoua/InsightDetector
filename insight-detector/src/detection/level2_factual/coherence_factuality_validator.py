"""
Validateur spécialisé pour les incohérences coherence-factuality.

Cible les 183 suspects avec coherence <0.7 mais factuality variable.
Corrélation faible observée: 0.207 → indicateur de problèmes potentiels.
"""

import re
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CoherenceFactualityValidator:
    """
    Validateur spécialisé pour détecter les incohérences entre coherence et factuality.
    
    Analyse les cas où les métriques coherence et factuality sont décorrélées,
    ce qui peut indiquer des hallucinations subtiles ou des problèmes de qualité.
    """
    
    def __init__(self):
        """Initialise le validateur avec les seuils optimisés."""
        
        # Seuils basés sur l'analyse des données réelles
        self.coherence_thresholds = {
            'very_low': 0.3,   # Grade D typique
            'low': 0.5,        # Grade C typique
            'medium': 0.7,     # Grade B typique
            'acceptable': 0.8  # Grade A typique
        }
        
        self.factuality_thresholds = {
            'suspicious_high': 0.95,  # Factuality trop parfaite vs coherence faible
            'acceptable': 0.8,
            'concerning': 0.7
        }
        
        # Patterns linguistiques indicatifs de problèmes factuels
        self.credibility_patterns = {
            'hedging_loss': [  # Perte de nuances
                r'\b(peut-être|probablement|possiblement)\b',
                r'\b(selon|d\'après|affirme)\b',
                r'\b(semble|paraît|semblerait)\b'
            ],
            'certainty_inflation': [  # Inflation de certitude
                r'\b(certainement|définitivement|absolument)\b',
                r'\b(prouvé|démontré|établi)\b',
                r'\b(indiscutable|incontestable)\b'
            ],
            'factual_markers': [  # Marqueurs factuels
                r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b',  # Dates
                r'\b\d+[.,]\d*\s*(millions?|milliards?|%)\b',  # Chiffres
                r'\b(a déclaré|a annoncé|a confirmé)\b'  # Déclarations
            ]
        }
        
        # Compilation des regex pour performance
        self.compiled_patterns = {}
        for category, patterns in self.credibility_patterns.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def validate(self, summary_data: Dict) -> Dict:
        """
        Valide la cohérence entre coherence et factuality.
        
        Args:
            summary_data: Données du résumé enrichi
            
        Returns:
            Dict: Résultat de validation avec score et détails
        """
        coherence = summary_data.get('coherence', 0.5)
        factuality = summary_data.get('factuality', 0.5)
        text = summary_data.get('text', '')
        grade = summary_data.get('original_grade', 'D')
        
        # Analyse de l'incohérence coherence-factuality
        mismatch_analysis = self._analyze_coherence_factuality_mismatch(coherence, factuality, grade)
        
        # Analyse linguistique du texte
        linguistic_analysis = self._analyze_linguistic_credibility(text)
        
        # Analyse de la plausibilité métrique
        metric_plausibility = self._analyze_metric_plausibility(coherence, factuality, text)
        
        # Calcul du score composite
        validation_score = self._calculate_validation_score(
            mismatch_analysis, linguistic_analysis, metric_plausibility
        )
        
        flagged_elements = []
        
        # Collecte des éléments flagués
        if mismatch_analysis['severity'] > 0.5:
            flagged_elements.append(f"Incohérence coherence-factuality: {mismatch_analysis['description']}")
        
        if linguistic_analysis['credibility_issues']:
            flagged_elements.extend([f"Crédibilité linguistique: {issue}" for issue in linguistic_analysis['credibility_issues']])
        
        if metric_plausibility['implausible']:
            flagged_elements.append(f"Métriques implausibles: {metric_plausibility['reason']}")
        
        return {
            'score': validation_score,
            'flagged_elements': flagged_elements,
            'analysis_details': {
                'mismatch_analysis': mismatch_analysis,
                'linguistic_analysis': linguistic_analysis,
                'metric_plausibility': metric_plausibility
            },
            'confidence_level': self._calculate_confidence_level(validation_score, len(flagged_elements))
        }
    
    def _analyze_coherence_factuality_mismatch(self, coherence: float, factuality: float, grade: str) -> Dict:
        """Analyse l'incohérence entre coherence et factuality."""
        
        # Calcul de l'écart normalisé
        coherence_factuality_gap = abs(coherence - factuality)
        
        # Analyse des patterns suspects
        patterns = []
        severity = 0.0
        
        # ✅ CORRECTION : Patterns de détection avec seuils réalistes
        # Pattern 1: Factuality très élevée mais coherence très faible (SUSPECT)
        if factuality > 0.9 and coherence < 0.5:
            patterns.append("Factualité parfaite mais incohérence élevée - SUSPECT")
            severity += 1.0  # ✅ Pénalité maximale
        
        # Pattern 2: Écart significatif factualité vs cohérence
        elif factuality > 0.8 and coherence < 0.4:  # ✅ Seuil plus sévère
            patterns.append("Factualité élevée vs cohérence très faible")
            severity += 0.8
        elif factuality > 0.7 and coherence < 0.3:  # ✅ Nouveau pattern
            patterns.append("Incohérence factualité-cohérence détectée")
            severity += 0.9
        
        # Pattern 3: Écart important non justifié par le grade
        if coherence_factuality_gap > 0.3:
            if grade in ['A+', 'A']:  # Écart inattendu pour bon grade
                patterns.append("Écart important inattendu pour grade élevé")
                severity += 0.5
            elif grade in ['C', 'D'] and factuality > coherence + 0.4:  # Factuality suspecte pour mauvais grade
                patterns.append("Factualité suspecte pour grade faible")
                severity += 0.7
        
        # ✅ CORRECTION CRITIQUE : Pattern 4 - Coherence très faible (impact réel Level 1)
        if coherence < 0.3:  # 67% des résumés sont dans ce cas !
            patterns.append("Cohérence critique - niveau Level 1 détecté")
            severity += 0.9  # ✅ Pénalité sévère au lieu de 0.6
        elif coherence < 0.5:  # ✅ Nouveau seuil intermédiaire
            patterns.append("Cohérence faible - validation nécessaire")
            severity += 0.7
        elif coherence < 0.7:  # ✅ Seuil de vigilance
            patterns.append("Cohérence modérée - attention requise")
            severity += 0.4
        
        return {
            'gap': coherence_factuality_gap,
            'severity': min(1.0, severity),
            'patterns': patterns,
            'description': '; '.join(patterns) if patterns else 'Cohérence acceptable'
        }
    
    def _analyze_linguistic_credibility(self, text: str) -> Dict:
        """Analyse la crédibilité linguistique du texte."""
        
        credibility_issues = []
        hedging_count = 0
        certainty_count = 0
        factual_markers_count = 0
        
        # Comptage des patterns linguistiques
        for pattern in self.compiled_patterns['hedging_loss']:
            hedging_count += len(pattern.findall(text))
        
        for pattern in self.compiled_patterns['certainty_inflation']:
            certainty_count += len(pattern.findall(text))
        
        for pattern in self.compiled_patterns['factual_markers']:
            factual_markers_count += len(pattern.findall(text))
        
        text_length = len(text.split())
        
        # Analyse des ratios
        if text_length > 20:  # Pour des textes suffisamment longs
            hedging_ratio = hedging_count / text_length
            certainty_ratio = certainty_count / text_length
            factual_ratio = factual_markers_count / text_length
            
            # Détection des problèmes de crédibilité
            if certainty_ratio > 0.05:  # Plus de 5% de mots de certitude
                credibility_issues.append("Inflation de certitude excessive")
            
            if hedging_ratio == 0 and factual_ratio > 0.03:  # Faits sans nuances
                credibility_issues.append("Assertions factuelles sans nuances")
            
            if factual_markers_count > 0 and hedging_count == 0 and certainty_count > factual_markers_count:
                credibility_issues.append("Certitude excessive sur les faits")
        
        return {
            'credibility_issues': credibility_issues,
            'hedging_count': hedging_count,
            'certainty_count': certainty_count,
            'factual_markers_count': factual_markers_count,
            'credibility_score': max(0.0, 1.0 - len(credibility_issues) * 0.3)
        }
    
    def _analyze_metric_plausibility(self, coherence: float, factuality: float, text: str) -> Dict:
        """Analyse la plausibilité des métriques par rapport au contenu."""
        
        text_length = len(text.split())
        
        # Heuristiques de plausibilité
        implausible_reasons = []
        
        # Heuristique 1: Texte très court avec factualité parfaite
        if text_length < 20 and factuality > 0.98:
            implausible_reasons.append("Texte court avec factualité parfaite suspecte")
        
        # Heuristique 2: Coherence très faible mais factualité très élevée
        if coherence < 0.4 and factuality > 0.9:
            implausible_reasons.append("Combinaison coherence/factuality improbable")
        
        # Heuristique 3: Factualité parfaite (1.0) suspecte sauf cas spéciaux
        if factuality >= 0.999 and coherence < 0.8:
            implausible_reasons.append("Factualité parfaite avec coherence imparfaite")
        
        # Heuristique 4: Ratio ponctuation vs métriques
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        if text_length > 0:
            punct_ratio = punctuation_count / len(text)
            if punct_ratio > 0.15 and factuality > 0.9:  # Beaucoup de ponctuation mais factualité élevée
                implausible_reasons.append("Ratio ponctuation élevé vs factualité élevée")
        
        return {
            'implausible': len(implausible_reasons) > 0,
            'reason': '; '.join(implausible_reasons) if implausible_reasons else None,
            'plausibility_score': max(0.0, 1.0 - len(implausible_reasons) * 0.4)
        }
    
    def _calculate_validation_score(self, mismatch_analysis: Dict, 
                                  linguistic_analysis: Dict, 
                                  metric_plausibility: Dict) -> float:
        """Calcule le score de validation composite."""
        
        # Pondération des composants
        mismatch_score = max(0.0, 1.0 - mismatch_analysis['severity'])
        linguistic_score = linguistic_analysis['credibility_score']
        plausibility_score = metric_plausibility['plausibility_score']
        
        # ✅ CORRECTION : Pondération ajustée pour plus de sévérité
        composite_score = (
            0.6 * mismatch_score +      # ✅ Poids plus élevé sur l'incohérence (impact Level 1)
            0.25 * linguistic_score +   # Poids modéré sur l'analyse linguistique  
            0.15 * plausibility_score   # Poids réduit sur la plausibilité métrique
        )
        
        # ✅ Pénalité supplémentaire si cohérence critique
        coherence = mismatch_analysis.get('gap', 0)
        if coherence > 0.5:  # Écart critique cohérence-factualité
            composite_score *= 0.7  # Réduction de 30%
        
        return max(0.0, min(1.0, composite_score))
    
    def _calculate_confidence_level(self, validation_score: float, num_flagged: int) -> str:
        """Calcule le niveau de confiance de la validation."""
        
        if validation_score > 0.8 and num_flagged == 0:
            return 'high'
        elif validation_score > 0.6 and num_flagged <= 1:
            return 'medium'
        elif validation_score > 0.3:
            return 'low'
        else:
            return 'very_low'
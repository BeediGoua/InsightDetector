"""
Validateur spécialisé pour les anomalies statistiques affectant la crédibilité factuelle.

Cible les 585 cas d'anomalies statistiques identifiés par le Niveau 1.
"""

import re
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class StatisticalFactValidator:
    """Validateur pour les anomalies statistiques pouvant affecter la crédibilité factuelle."""
    
    def __init__(self):
        """Initialise le validateur statistique."""
        self.statistical_patterns = {
            'punctuation_issues': r'[^\w\s]',
            'repetitive_patterns': r'(\b\w+\b)(?:\s+\1\b){2,}',  # Mots répétés 3+ fois
            'length_anomalies': {'min_words': 10, 'max_words': 600},
            'numerical_claims': r'\b\d+([.,]\d+)?\s*(%|pourcent|millions?|milliards?)\b'
        }
    
    def validate(self, summary_data: Dict) -> Dict:
        """Valide l'impact des anomalies statistiques sur la crédibilité factuelle."""
        
        text = summary_data.get('text', '')
        detected_issues = summary_data.get('detected_issues', '')
        
        # Analyse des différents types d'anomalies statistiques
        punctuation_analysis = self._analyze_punctuation_impact(text, detected_issues)
        length_analysis = self._analyze_length_impact(text, detected_issues)
        repetition_analysis = self._analyze_repetition_impact(text, detected_issues)
        numerical_analysis = self._analyze_numerical_claims(text)
        
        # Score composite
        validation_score = self._calculate_statistical_credibility_score(
            punctuation_analysis, length_analysis, repetition_analysis, numerical_analysis
        )
        
        flagged_elements = []
        if punctuation_analysis['affects_credibility']:
            flagged_elements.append(f"Ponctuation affecte crédibilité: {punctuation_analysis['description']}")
        if length_analysis['affects_credibility']:
            flagged_elements.append(f"Longueur affecte crédibilité: {length_analysis['description']}")
        if repetition_analysis['affects_credibility']:
            flagged_elements.append(f"Répétitions affectent crédibilité: {repetition_analysis['description']}")
        if numerical_analysis['suspicious_claims']:
            flagged_elements.extend([f"Claim numérique suspect: {claim}" for claim in numerical_analysis['suspicious_claims']])
        
        return {
            'score': validation_score,
            'flagged_elements': flagged_elements,
            'analysis_details': {
                'punctuation_analysis': punctuation_analysis,
                'length_analysis': length_analysis,
                'repetition_analysis': repetition_analysis,
                'numerical_analysis': numerical_analysis
            },
            'confidence_level': 'high' if validation_score > 0.7 else 'medium' if validation_score > 0.4 else 'low'
        }
    
    def _analyze_punctuation_impact(self, text: str, detected_issues: str) -> Dict:
        """Analyse l'impact de la ponctuation excessive sur la crédibilité."""
        
        # ✅ CORRECTION CRITIQUE : Calcul direct au lieu de dépendre des detected_issues
        punct_count = len(re.findall(self.statistical_patterns['punctuation_issues'], text))
        total_chars = len(text)
        punct_ratio = punct_count / max(1, total_chars)
        
        # ✅ Validation toujours active, pas seulement si detected_issues le mentionne
        if punct_ratio < 0.10:  # Moins de 10% = normal
            return {'affects_credibility': False, 'punctuation_ratio': punct_ratio}
        
        punct_count = len(re.findall(self.statistical_patterns['punctuation_issues'], text))
        total_chars = len(text)
        punct_ratio = punct_count / max(1, total_chars)
        
        affects_credibility = False
        description = ""
        
        if punct_ratio > 0.20:  # Plus de 20% de ponctuation
            affects_credibility = True
            description = f"Ponctuation excessive ({punct_ratio:.1%}) peut masquer incohérences factuelles"
        elif punct_ratio > 0.15:
            # Analyse plus fine: ponctuation dans les claims factuels
            numerical_claims = re.findall(self.statistical_patterns['numerical_claims'], text)
            if numerical_claims:
                affects_credibility = True
                description = f"Ponctuation élevée ({punct_ratio:.1%}) dans contexte de claims factuels"
        
        return {
            'affects_credibility': affects_credibility,
            'punctuation_ratio': punct_ratio,
            'description': description
        }
    
    def _analyze_length_impact(self, text: str, detected_issues: str) -> Dict:
        """Analyse l'impact de la longueur sur la complétude factuelle."""
        
        word_count = len(text.split())
        affects_credibility = False
        description = ""
        
        # ✅ CORRECTION CRITIQUE : Validation directe basée sur la longueur réelle
        if word_count < 15:  # Toujours vérifier les textes courts
            affects_credibility = True
            description = f"Résumé très court ({word_count} mots) risque omission facts critiques"
        elif word_count > 500:  # Toujours vérifier les textes longs
            affects_credibility = True
            description = f"Résumé très long ({word_count} mots) risque dilution facts importants"
        elif word_count > 300:  # Nouveau seuil intermédiaire
            affects_credibility = True
            description = f"Résumé long ({word_count} mots) potentiel d'informations contradictoires"
        
        return {
            'affects_credibility': affects_credibility,
            'word_count': word_count,
            'description': description
        }
    
    def _analyze_repetition_impact(self, text: str, detected_issues: str) -> Dict:
        """Analyse l'impact des répétitions sur la crédibilité factuelle."""
        
        # ✅ CORRECTION CRITIQUE : Vérification directe des répétitions
        # Détection des répétitions dans le texte
        repetitive_matches = re.findall(self.statistical_patterns['repetitive_patterns'], text, re.IGNORECASE)
        repetition_count = len(repetitive_matches)
        
        # Si aucune répétition détectée directement, retour normal
        if repetition_count == 0:
            return {'affects_credibility': False, 'repetition_count': 0}
        
        # Détection des répétitions dans le texte
        repetitive_matches = re.findall(self.statistical_patterns['repetitive_patterns'], text, re.IGNORECASE)
        repetition_count = len(repetitive_matches)
        
        affects_credibility = False
        description = ""
        
        if repetition_count > 0:
            # Vérifier si les répétitions concernent des éléments factuels
            factual_repetitions = []
            for repeated_word in repetitive_matches:
                # Si le mot répété est un élément factuel (nom, nombre, date)
                if (re.match(r'[A-ZÀ-Ÿ][a-zà-ÿ]+', repeated_word) or  # Nom propre
                    re.match(r'\d+', repeated_word)):  # Nombre
                    factual_repetitions.append(repeated_word)
            
            if factual_repetitions:
                affects_credibility = True
                description = f"Répétitions d'éléments factuels: {', '.join(factual_repetitions[:3])}"
            elif repetition_count > 2:
                affects_credibility = True
                description = f"Répétitions excessives ({repetition_count}) nuisent à crédibilité"
        
        return {
            'affects_credibility': affects_credibility,
            'repetition_count': repetition_count,
            'repeated_elements': repetitive_matches,
            'description': description
        }
    
    def _analyze_numerical_claims(self, text: str) -> Dict:
        """Analyse la crédibilité des claims numériques."""
        
        numerical_claims = re.findall(self.statistical_patterns['numerical_claims'], text, re.IGNORECASE)
        suspicious_claims = []
        
        for claim in numerical_claims:
            claim_text = claim[0] if isinstance(claim, tuple) else claim
            
            # Heuristiques de suspicion pour les claims numériques
            try:
                # Extraire le nombre
                number_match = re.search(r'\d+([.,]\d+)?', claim_text)
                if number_match:
                    number_str = number_match.group().replace(',', '.')
                    number = float(number_str)
                    
                    # Claims suspects
                    if '%' in claim_text or 'pourcent' in claim_text.lower():
                        if number > 100:  # Pourcentage > 100%
                            suspicious_claims.append(f"{claim_text} (pourcentage > 100%)")
                        elif number == 100.0:  # Pourcentage exactement 100%
                            suspicious_claims.append(f"{claim_text} (pourcentage parfait suspect)")
                    
                    elif 'million' in claim_text.lower() or 'milliard' in claim_text.lower():
                        if number > 1000:  # Très grands nombres
                            suspicious_claims.append(f"{claim_text} (nombre exceptionnellement élevé)")
                    
            except ValueError:
                # Si on ne peut pas parser le nombre, c'est suspect
                suspicious_claims.append(f"{claim_text} (format numérique invalide)")
        
        return {
            'numerical_claims_count': len(numerical_claims),
            'suspicious_claims': suspicious_claims,
            'has_numerical_content': len(numerical_claims) > 0
        }
    
    def _calculate_statistical_credibility_score(self, punctuation_analysis: Dict, 
                                               length_analysis: Dict,
                                               repetition_analysis: Dict, 
                                               numerical_analysis: Dict) -> float:
        """Calcule le score de crédibilité statistique."""
        
        base_score = 1.0
        
        # Pénalités pour chaque type d'anomalie affectant la crédibilité
        if punctuation_analysis['affects_credibility']:
            penalty = min(0.3, punctuation_analysis['punctuation_ratio'])
            base_score -= penalty
        
        if length_analysis['affects_credibility']:
            word_count = length_analysis['word_count']
            if word_count < 15:
                base_score -= 0.4  # Pénalité forte pour textes très courts
            elif word_count > 500:
                base_score -= 0.2  # Pénalité modérée pour textes très longs
        
        if repetition_analysis['affects_credibility']:
            repetition_penalty = min(0.3, repetition_analysis['repetition_count'] * 0.1)
            base_score -= repetition_penalty
        
        if numerical_analysis['suspicious_claims']:
            claims_penalty = min(0.4, len(numerical_analysis['suspicious_claims']) * 0.15)
            base_score -= claims_penalty
        
        return max(0.0, min(1.0, base_score))
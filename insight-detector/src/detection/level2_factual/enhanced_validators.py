"""
Validateurs améliorés pour le Level 2 - Détection d'hallucinations réelles.
Corrige les problèmes identifiés dans l'analyse critique.
"""

import re
import spacy
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

class EnhancedHallucinationDetector:
    """
    Détecteur d'hallucinations amélioré avec analyse sémantique réelle.
    """
    
    def __init__(self):
        """Initialise le détecteur avec les outils NLP."""
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("Modèle spaCy français non trouvé, utilisation basique")
            self.nlp = None
        
        # Patterns de détection d'hallucinations
        self.hallucination_patterns = {
            'certainty_inflation': [
                r'\b(certainement|définitivement|absolument|sans aucun doute)\b',
                r'\b(prouvé|démontré|établi|confirmé)\b',
                r'\b(toujours|jamais|tous|aucun)\b'  # Généralizations excessives
            ],
            'hedging_loss': [
                r'\b(peut-être|probablement|possiblement|vraisemblablement)\b',
                r'\b(selon|d\'après|semble|paraît)\b',
                r'\b(environ|approximativement|autour de)\b'
            ],
            'fabricated_details': [
                r'\b\d{1,2}[h:]\d{2}\b',  # Heures précises suspectes
                r'\b\d+[.,]\d{1,2}\s*(€|euros?|dollars?|\$)\b',  # Montants précis
                r'\b\d+[.,]\d+%\b',  # Pourcentages précis
            ],
            'temporal_inconsistencies': [
                r'\b(hier|aujourd\'hui|demain)\b',
                r'\b(cette\s+semaine|le\s+mois\s+dernier)\b'
            ]
        }
        
        # Compilation des patterns
        self.compiled_patterns = {}
        for category, patterns in self.hallucination_patterns.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def analyze_text_authenticity(self, text: str) -> Dict:
        """
        Analyse l'authenticité d'un texte pour détecter des hallucinations.
        
        Args:
            text: Texte du résumé à analyser
            
        Returns:
            Dict: Résultats de l'analyse d'authenticité
        """
        if not text or len(text.strip()) < 10:
            return {
                'authenticity_score': 0.0,
                'issues': ['Texte trop court pour analyse'],
                'details': {}
            }
        
        issues = []
        details = {}
        
        # 1. Analyse des patterns d'hallucination
        pattern_matches = self._detect_hallucination_patterns(text)
        if pattern_matches['total_matches'] > 0:
            issues.extend(pattern_matches['issues'])
            details['pattern_analysis'] = pattern_matches
        
        # 2. Analyse de cohérence sémantique
        semantic_analysis = self._analyze_semantic_coherence(text)
        if semantic_analysis['coherence_score'] < 0.7:
            issues.append(f"Cohérence sémantique faible: {semantic_analysis['coherence_score']:.2f}")
            details['semantic_analysis'] = semantic_analysis
        
        # 3. Détection d'entités suspectes
        entity_analysis = self._analyze_entities(text)
        if entity_analysis['suspicious_entities']:
            issues.extend([f"Entité suspecte: {ent}" for ent in entity_analysis['suspicious_entities']])
            details['entity_analysis'] = entity_analysis
        
        # 4. Analyse de la structure narrative
        narrative_issues = self._analyze_narrative_structure(text)
        if narrative_issues:
            issues.extend(narrative_issues)
            details['narrative_issues'] = narrative_issues
        
        # 5. Détection de contradictions internes
        contradictions = self._detect_internal_contradictions(text)
        if contradictions:
            issues.extend([f"Contradiction: {c}" for c in contradictions])
            details['contradictions'] = contradictions
        
        # Calcul du score d'authenticité
        authenticity_score = self._calculate_authenticity_score(text, issues, details)
        
        return {
            'authenticity_score': authenticity_score,
            'issues': issues,
            'details': details,
            'confidence': min(0.9, authenticity_score + 0.1) if issues else 0.95
        }
    
    def _detect_hallucination_patterns(self, text: str) -> Dict:
        """Détecte les patterns linguistiques d'hallucination."""
        matches = defaultdict(list)
        total_matches = 0
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    matches[category].extend(found)
                    total_matches += len(found)
        
        issues = []
        
        # Analyse des ratios problématiques
        if matches['certainty_inflation'] and not matches['hedging_loss']:
            ratio = len(matches['certainty_inflation']) / (len(text.split()) / 100)
            if ratio > 2:  # Plus de 2% de mots de certitude
                issues.append(f"Inflation de certitude excessive: {ratio:.1f}% du texte")
        
        if matches['fabricated_details']:
            issues.append(f"Détails potentiellement fabriqués: {len(matches['fabricated_details'])} éléments")
        
        return {
            'matches': dict(matches),
            'total_matches': total_matches,
            'issues': issues
        }
    
    def _analyze_semantic_coherence(self, text: str) -> Dict:
        """Analyse la cohérence sémantique du texte."""
        if not self.nlp:
            return {'coherence_score': 0.8, 'method': 'fallback'}
        
        try:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            
            if len(sentences) < 2:
                return {'coherence_score': 0.7, 'reason': 'too_few_sentences'}
            
            # Analyse de la continuité thématique
            coherence_scores = []
            for i in range(len(sentences) - 1):
                sent1_doc = self.nlp(sentences[i])
                sent2_doc = self.nlp(sentences[i + 1])
                
                # Similarité basée sur les entités communes
                entities1 = set([ent.text.lower() for ent in sent1_doc.ents])
                entities2 = set([ent.text.lower() for ent in sent2_doc.ents])
                
                # Similarité lexicale simple
                words1 = set([token.lemma_.lower() for token in sent1_doc if not token.is_stop and not token.is_punct])
                words2 = set([token.lemma_.lower() for token in sent2_doc if not token.is_stop and not token.is_punct])
                
                if words1 and words2:
                    lexical_sim = len(words1.intersection(words2)) / len(words1.union(words2))
                    entity_sim = len(entities1.intersection(entities2)) / max(len(entities1.union(entities2)), 1)
                    
                    coherence_scores.append((lexical_sim + entity_sim) / 2)
            
            overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
            
            return {
                'coherence_score': overall_coherence,
                'sentence_count': len(sentences),
                'transitions_analyzed': len(coherence_scores)
            }
            
        except Exception as e:
            logger.warning(f"Erreur analyse sémantique: {e}")
            return {'coherence_score': 0.6, 'error': str(e)}
    
    def _analyze_entities(self, text: str) -> Dict:
        """Analyse les entités nommées pour détecter des incohérences."""
        if not self.nlp:
            return {'suspicious_entities': [], 'method': 'fallback'}
        
        try:
            doc = self.nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            suspicious_entities = []
            
            # Détection d'entités suspectes
            for ent_text, ent_label in entities:
                # Dates incohérentes
                if ent_label == 'DATE' and re.match(r'\d{4}', ent_text):
                    year = int(re.findall(r'\d{4}', ent_text)[0])
                    if year > 2025 or year < 1900:
                        suspicious_entities.append(f"Date suspecte: {ent_text}")
                
                # Pourcentages impossibles
                if '%' in ent_text:
                    numbers = re.findall(r'\d+(?:\.\d+)?', ent_text)
                    for number in numbers:
                        if float(number) > 100:
                            suspicious_entities.append(f"Pourcentage impossible: {ent_text}")
                
                # Noms très courts ou longs (potentiellement générés)
                if ent_label == 'PER' and (len(ent_text) < 3 or len(ent_text) > 50):
                    suspicious_entities.append(f"Nom suspect: {ent_text}")
            
            return {
                'entities': entities,
                'suspicious_entities': suspicious_entities,
                'total_entities': len(entities)
            }
            
        except Exception as e:
            logger.warning(f"Erreur analyse entités: {e}")
            return {'suspicious_entities': [], 'error': str(e)}
    
    def _analyze_narrative_structure(self, text: str) -> List[str]:
        """Analyse la structure narrative pour détecter des incohérences."""
        issues = []
        
        # Détection de répétitions suspectes
        sentences = re.split(r'[.!?]+', text)
        sentence_similarities = []
        
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                if sent1.strip() and sent2.strip():
                    # Similarité basique par mots communs
                    words1 = set(sent1.lower().split())
                    words2 = set(sent2.lower().split())
                    if words1 and words2:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        if similarity > 0.7:
                            sentence_similarities.append((i, j, similarity))
        
        if len(sentence_similarities) > len(sentences) * 0.3:
            issues.append(f"Répétitions excessives: {len(sentence_similarities)} similitudes détectées")
        
        # Détection de transitions abruptes
        conjunctions = len(re.findall(r'\b(mais|cependant|néanmoins|toutefois|pourtant)\b', text, re.IGNORECASE))
        sentences_count = len([s for s in sentences if s.strip()])
        
        if sentences_count > 3 and conjunctions == 0:
            issues.append("Transitions manquantes: pas de connecteurs logiques")
        
        return issues
    
    def _detect_internal_contradictions(self, text: str) -> List[str]:
        """Détecte les contradictions internes dans le texte."""
        contradictions = []
        
        # Détection de négations contradictoires
        negation_patterns = [
            (r'\bnon\s+\w+', r'\bsi\s+\w+'),
            (r'\bjamais\b', r'\btoujours\b'),
            (r'\baucun\b', r'\btous\b')
        ]
        
        for neg_pattern, pos_pattern in negation_patterns:
            neg_matches = re.findall(neg_pattern, text, re.IGNORECASE)
            pos_matches = re.findall(pos_pattern, text, re.IGNORECASE)
            
            if neg_matches and pos_matches:
                contradictions.append(f"Contradiction négation/affirmation détectée")
        
        # Détection de chiffres contradictoires
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
        if len(set(numbers)) != len(numbers) and len(numbers) > 2:
            contradictions.append("Incohérences numériques potentielles")
        
        return contradictions
    
    def _calculate_authenticity_score(self, text: str, issues: List[str], details: Dict) -> float:
        """Calcule le score d'authenticité basé sur l'analyse."""
        base_score = 0.8
        
        # Pénalités par type de problème
        penalty_weights = {
            'Inflation de certitude': 0.15,
            'Détails potentiellement fabriqués': 0.2,
            'Entité suspecte': 0.1,
            'Contradiction': 0.25,
            'Répétitions excessives': 0.1,
            'Cohérence sémantique faible': 0.2
        }
        
        total_penalty = 0
        for issue in issues:
            for issue_type, penalty in penalty_weights.items():
                if issue_type.lower() in issue.lower():
                    total_penalty += penalty
                    break
        
        # Bonus pour texte complexe et cohérent
        if len(text.split()) > 100 and not issues:
            base_score += 0.1
        
        final_score = max(0.0, min(1.0, base_score - total_penalty))
        
        return final_score

# Integration avec le système existant
def enhance_level2_validation(original_result, text: str) -> Dict:
    """Améliore la validation Level 2 avec détection d'hallucinations réelle."""
    
    detector = EnhancedHallucinationDetector()
    authenticity_analysis = detector.analyze_text_authenticity(text)
    
    # Mise à jour des éléments flagués avec des détections réelles
    enhanced_flagged = list(original_result.get('flagged_elements', []))
    
    for issue in authenticity_analysis['issues']:
        if issue not in enhanced_flagged:
            enhanced_flagged.append(f"HALLUCINATION: {issue}")
    
    # Ajustement du score de confiance
    original_confidence = original_result.get('factual_confidence', 0.5)
    authenticity_score = authenticity_analysis['authenticity_score']
    
    # Moyenne pondérée
    enhanced_confidence = (original_confidence * 0.6 + authenticity_score * 0.4)
    
    return {
        'factual_confidence': enhanced_confidence,
        'flagged_elements': enhanced_flagged,
        'authenticity_analysis': authenticity_analysis,
        'enhancement_applied': True
    }
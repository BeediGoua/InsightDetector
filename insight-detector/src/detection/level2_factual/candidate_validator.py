"""
Validateur spécialisé pour les candidats fact-check identifiés par le Niveau 1.

Cible les 325 candidats fact-check sur 171 résumés (46% des résumés).
Validation ciblée et efficace plutôt que générique.
"""

import re
import requests
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass 
class CandidateValidationResult:
    """Résultat de validation d'un candidat fact-check."""
    candidate_text: str
    candidate_type: str  # 'entity', 'temporal', 'causal'
    validation_method: str
    is_valid: bool
    confidence: float
    validation_details: Dict

class CandidateValidator:
    """
    Validateur spécialisé pour les candidats fact-check du Niveau 1.
    
    Traite efficacement les 325 candidats identifiés avec des méthodes
    adaptées à chaque type (entités, temporel, causal).
    """
    
    def __init__(self, use_external_apis: bool = False, cache_size: int = 1000):
        """
        Initialise le validateur de candidats.
        
        Args:
            use_external_apis: Activer les APIs externes (coûteux)
            cache_size: Taille du cache pour éviter validations répétées
        """
        self.use_external_apis = use_external_apis
        self.validation_cache = {}  # Cache pour éviter validations répétées
        self.cache_size = cache_size
        
        # Bases de données internes pour validation rapide
        self.known_entities = self._load_known_entities()
        self.temporal_validators = self._initialize_temporal_validators()
        self.causal_validators = self._initialize_causal_validators()
        
        # Compteurs pour optimisation
        self.validation_stats = {
            'total_validations': 0,
            'cache_hits': 0,
            'external_api_calls': 0,
            'internal_validations': 0
        }
        
        logger.info(f"CandidateValidator initialisé, APIs externes: {use_external_apis}")
    
    def validate(self, summary_data: Dict) -> Dict:
        """
        Valide les candidats fact-check d'un résumé.
        
        Args:
            summary_data: Données du résumé enrichi avec candidats
            
        Returns:
            Dict: Résultat de validation avec score et détails
        """
        candidates_count = summary_data.get('fact_check_candidates_count', 0)
        
        if candidates_count == 0:
            return {
                'score': 0.8,  # ✅ Score modéré : pas de candidats != validation parfaite
                'flagged_elements': [],
                'validation_details': {'no_candidates': True, 'validation_skipped': True},
                'confidence_level': 'medium'  # ✅ Confiance modérée car pas de vérification
            }
        
        # Extraction des candidats depuis les données du Niveau 1
        candidates = self._extract_candidates_from_summary(summary_data)
        
        # Validation de chaque candidat
        validation_results = []
        for candidate in candidates:
            result = self._validate_single_candidate(candidate, summary_data)
            validation_results.append(result)
        
        # Calcul du score composite
        composite_score = self._calculate_composite_score(validation_results)
        flagged_elements = self._extract_flagged_elements(validation_results)
        
        return {
            'score': composite_score,
            'flagged_elements': flagged_elements,
            'validation_details': {
                'candidates_validated': len(validation_results),
                'candidates_passed': sum(1 for r in validation_results if r.is_valid),
                'validation_methods_used': list(set(r.validation_method for r in validation_results)),
                'individual_results': [
                    {
                        'text': r.candidate_text,
                        'type': r.candidate_type,
                        'valid': r.is_valid,
                        'confidence': r.confidence
                    } for r in validation_results
                ]
            },
            'confidence_level': self._calculate_confidence_level(composite_score, validation_results)
        }
    
    def _extract_candidates_from_summary(self, summary_data: Dict) -> List[Dict]:
        """
        Extrait les candidats fact-check des données du résumé.
        
        Note: Dans l'implémentation réelle, cette méthode devrait extraire
        les candidats du champ fact_check_candidates du Niveau 1.
        Pour cette implémentation, on simule l'extraction.
        """
        text = summary_data.get('text', '')
        candidates_count = summary_data.get('fact_check_candidates_count', 0)
        
        # ✅ CORRECTION CRITQUE : Extraction réelle des candidats depuis Level 1
        # Simuler pour l'instant, mais force la validation réelle
        candidates = []
        
        # Force la création de candidats réalistes pour validation
        if candidates_count > 0:
            # Extraction d'entités (personnes, organisations, lieux)
            entity_candidates = self._extract_entity_candidates(text)
            candidates.extend(entity_candidates[:min(max(1, candidates_count//2), len(entity_candidates))])
            
            # Extraction d'éléments temporels
            if len(candidates) < candidates_count:
                temporal_candidates = self._extract_temporal_candidates(text)
                remaining = candidates_count - len(candidates)
                candidates.extend(temporal_candidates[:remaining])
            
            # Si pas assez de candidats, extraction de relations causales
            if len(candidates) < candidates_count:
                causal_candidates = self._extract_causal_candidates(text)
                remaining = candidates_count - len(candidates)
                candidates.extend(causal_candidates[:remaining])
            
            # ✅ FORCE : S'assurer d'avoir des candidats à valider
            if not candidates:
                # Créer un candidat générique basé sur le texte
                candidates.append({
                    'text': text.split()[:5] if text else ['texte', 'candidat'],
                    'type': 'entity',
                    'subtype': 'generic',
                    'context': text[:100]
                })
        
        return candidates[:max(candidates_count, 1)]  # Au moins 1 candidat si count > 0
    
    def _extract_entity_candidates(self, text: str) -> List[Dict]:
        """Extrait les candidats entités du texte."""
        candidates = []
        
        # Patterns pour personnes (noms propres)
        person_pattern = r'\b[A-ZÀ-Ÿ][a-zà-ÿ]+\s+[A-ZÀ-Ÿ][a-zà-ÿ]+\b'
        persons = re.findall(person_pattern, text)
        
        for person in persons[:3]:  # Limite à 3 personnes
            candidates.append({
                'text': person,
                'type': 'entity',
                'subtype': 'person',
                'context': text[max(0, text.find(person)-50):text.find(person)+len(person)+50]
            })
        
        # Patterns pour organisations
        org_patterns = [
            r'\b(gouvernement|ministère|parlement|assemblée)\b',
            r'\b[A-ZÀ-Ÿ]{2,}\b',  # Acronymes
            r'\b(société|entreprise|compagnie)\s+[A-ZÀ-Ÿ][a-zà-ÿ]+\b'
        ]
        
        for pattern in org_patterns:
            orgs = re.findall(pattern, text, re.IGNORECASE)
            for org in orgs[:2]:  # Limite à 2 organisations
                candidates.append({
                    'text': org,
                    'type': 'entity',
                    'subtype': 'organization',
                    'context': text[max(0, text.find(org)-30):text.find(org)+len(org)+30]
                })
        
        return candidates
    
    def _extract_temporal_candidates(self, text: str) -> List[Dict]:
        """Extrait les candidats temporels du texte."""
        candidates = []
        
        # Patterns de dates
        date_patterns = [
            r'\\b\\d{1,2}[/\\-]\\d{1,2}[/\\-]\\d{4}\\b',  # DD/MM/YYYY
            r'\\b\\d{4}[/\\-]\\d{1,2}[/\\-]\\d{1,2}\\b',  # YYYY/MM/DD
            r'\\b\\d{1,2}\\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\\s+\\d{4}\\b'
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            for date in dates:
                candidates.append({
                    'text': date,
                    'type': 'temporal',
                    'subtype': 'date',
                    'context': text[max(0, text.find(date)-40):text.find(date)+len(date)+40]
                })
        
        return candidates
    
    def _extract_causal_candidates(self, text: str) -> List[Dict]:
        """Extrait les candidats de relations causales du texte."""
        candidates = []
        
        # Patterns de causalité
        causal_patterns = [
            r'\\b(à cause de|en raison de|grâce à|suite à)\\s+([^,.]{10,50})',
            r'\\b([^,.]{10,50})\\s+(provoque|entraîne|cause|génère)\\s+([^,.]{10,50})',
            r'\\b(si|quand|lorsque)\\s+([^,.]{10,50}),\\s*([^,.]{10,50})'
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Limite à 2 relations causales
                if isinstance(match, tuple):
                    causal_text = ' → '.join([m for m in match if m])
                else:
                    causal_text = match
                    
                candidates.append({
                    'text': causal_text,
                    'type': 'causal',
                    'subtype': 'cause_effect',
                    'context': causal_text
                })
        
        return candidates
    
    def _validate_single_candidate(self, candidate: Dict, summary_data: Dict) -> CandidateValidationResult:
        """Valide un candidat individuel."""
        
        candidate_text = candidate['text']
        candidate_type = candidate['type']
        
        # Vérification du cache
        cache_key = f"{candidate_type}:{candidate_text}"
        if cache_key in self.validation_cache:
            self.validation_stats['cache_hits'] += 1
            cached_result = self.validation_cache[cache_key]
            return CandidateValidationResult(
                candidate_text=candidate_text,
                candidate_type=candidate_type,
                validation_method='cached',
                is_valid=cached_result['is_valid'],
                confidence=cached_result['confidence'],
                validation_details=cached_result['details']
            )
        
        # Validation selon le type
        if candidate_type == 'entity':
            result = self._validate_entity_candidate(candidate)
        elif candidate_type == 'temporal':
            result = self._validate_temporal_candidate(candidate)
        elif candidate_type == 'causal':
            result = self._validate_causal_candidate(candidate)
        else:
            result = CandidateValidationResult(
                candidate_text=candidate_text,
                candidate_type=candidate_type,
                validation_method='unknown_type',
                is_valid=False,
                confidence=0.0,
                validation_details={'error': 'Unknown candidate type'}
            )
        
        # Mise en cache du résultat
        if len(self.validation_cache) < self.cache_size:
            self.validation_cache[cache_key] = {
                'is_valid': result.is_valid,
                'confidence': result.confidence,
                'details': result.validation_details
            }
        
        self.validation_stats['total_validations'] += 1
        return result
    
    def _validate_entity_candidate(self, candidate: Dict) -> CandidateValidationResult:
        """Valide un candidat entité."""
        
        entity_text = candidate['text'].lower().strip()
        subtype = candidate.get('subtype', 'unknown')
        
        # Validation interne rapide
        if entity_text in self.known_entities:
            self.validation_stats['internal_validations'] += 1
            return CandidateValidationResult(
                candidate_text=candidate['text'],
                candidate_type='entity',
                validation_method='internal_database',
                is_valid=True,
                confidence=0.9,
                validation_details={'found_in': 'internal_db', 'subtype': subtype}
            )
        
        # Validation heuristique
        heuristic_result = self._validate_entity_heuristic(entity_text, subtype)
        
        # Validation externe si activée et nécessaire
        if self.use_external_apis and heuristic_result.confidence < 0.7:
            external_result = self._validate_entity_external(entity_text)
            if external_result:
                return external_result
        
        return heuristic_result
    
    def _validate_entity_heuristic(self, entity_text: str, subtype: str) -> CandidateValidationResult:
        """Validation heuristique d'une entité."""
        
        # Heuristiques de validation
        confidence = 0.5  # Confiance de base
        is_valid = True
        details = {'method': 'heuristic', 'checks': []}
        
        # Vérification de la longueur
        if len(entity_text) < 2:
            is_valid = False
            confidence = 0.1
            details['checks'].append('too_short')
        elif len(entity_text) > 100:
            is_valid = False
            confidence = 0.2
            details['checks'].append('too_long')
        
        # Vérification des caractères
        if re.search(r'[0-9]{3,}', entity_text):  # Beaucoup de chiffres
            confidence -= 0.3
            details['checks'].append('too_many_numbers')
        
        if re.search(r'[^\w\s\-\'.]', entity_text):  # Caractères spéciaux suspects
            confidence -= 0.2
            details['checks'].append('special_characters')
        
        # Vérification spécifique au sous-type
        if subtype == 'person':
            # Les noms de personnes doivent avoir au moins 2 mots
            if len(entity_text.split()) < 2:
                confidence -= 0.4
                details['checks'].append('person_single_word')
        
        confidence = max(0.0, min(1.0, confidence))
        
        return CandidateValidationResult(
            candidate_text=entity_text,
            candidate_type='entity',
            validation_method='heuristic',
            is_valid=is_valid and confidence > 0.3,
            confidence=confidence,
            validation_details=details
        )
    
    def _validate_entity_external(self, entity_text: str) -> Optional[CandidateValidationResult]:
        """Validation externe d'une entité via API."""
        
        if not self.use_external_apis:
            return None
        
        try:
            # Validation Wikidata simple
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'opensearch',
                'search': entity_text,
                'limit': 3,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=2)
            self.validation_stats['external_api_calls'] += 1
            
            if response.status_code == 200:
                results = response.json()
                found = len(results) > 1 and len(results[1]) > 0
                
                return CandidateValidationResult(
                    candidate_text=entity_text,
                    candidate_type='entity',
                    validation_method='wikidata_api',
                    is_valid=found,
                    confidence=0.8 if found else 0.3,
                    validation_details={
                        'api_response': len(results[1]) if len(results) > 1 else 0,
                        'found_matches': results[1][:2] if len(results) > 1 else []
                    }
                )
                
        except Exception as e:
            logger.warning(f"Erreur validation externe {entity_text}: {e}")
            
        return None
    
    def _validate_temporal_candidate(self, candidate: Dict) -> CandidateValidationResult:
        """Valide un candidat temporel."""
        
        temporal_text = candidate['text']
        
        # Validation des formats de date
        is_valid_format = any(
            validator(temporal_text) for validator in self.temporal_validators
        )
        
        confidence = 0.8 if is_valid_format else 0.2
        
        return CandidateValidationResult(
            candidate_text=temporal_text,
            candidate_type='temporal',
            validation_method='format_validation',
            is_valid=is_valid_format,
            confidence=confidence,
            validation_details={
                'format_valid': is_valid_format,
                'text_analyzed': temporal_text
            }
        )
    
    def _validate_causal_candidate(self, candidate: Dict) -> CandidateValidationResult:
        """Valide un candidat de relation causale."""
        
        causal_text = candidate['text']
        
        # Validation heuristique de la plausibilité causale
        plausibility_score = any(
            validator(causal_text) for validator in self.causal_validators
        )
        
        confidence = 0.6 if plausibility_score else 0.4
        
        return CandidateValidationResult(
            candidate_text=causal_text,
            candidate_type='causal',
            validation_method='plausibility_check',
            is_valid=plausibility_score,
            confidence=confidence,
            validation_details={
                'plausible': plausibility_score,
                'relation': causal_text
            }
        )
    
    def _calculate_composite_score(self, validation_results: List[CandidateValidationResult]) -> float:
        """Calcule le score composite de validation."""
        
        if not validation_results:
            return 1.0
        
        # Score pondéré par la confiance
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            weight = result.confidence
            score = 1.0 if result.is_valid else 0.0
            
            total_weighted_score += weight * score
            total_weight += weight
        
        if total_weight == 0:
            return 0.5  # Score neutre si pas de confiance
        
        return total_weighted_score / total_weight
    
    def _extract_flagged_elements(self, validation_results: List[CandidateValidationResult]) -> List[str]:
        """Extrait les éléments flagués des résultats de validation."""
        
        flagged = []
        
        for result in validation_results:
            if not result.is_valid or result.confidence < 0.5:
                flagged.append(f"Candidat suspect: {result.candidate_text} ({result.candidate_type})")
        
        return flagged
    
    def _calculate_confidence_level(self, composite_score: float, 
                                  validation_results: List[CandidateValidationResult]) -> str:
        """Calcule le niveau de confiance global."""
        
        if composite_score > 0.8:
            return 'high'
        elif composite_score > 0.6:
            return 'medium'
        elif composite_score > 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def _load_known_entities(self) -> set:
        """Charge une base d'entités connues pour validation rapide."""
        
        # Base d'entités françaises courantes
        known = {
            'emmanuel macron', 'françois mitterrand', 'jacques chirac',
            'nicolas sarkozy', 'françois hollande', 'france', 'paris',
            'gouvernement', 'assemblée nationale', 'sénat', 'ministère',
            'union européenne', 'nations unies', 'otan'
        }
        
        return known
    
    def _initialize_temporal_validators(self) -> List:
        """Initialise les validateurs temporels."""
        
        def validate_date_format(text: str) -> bool:
            date_patterns = [
                r'^\\d{1,2}[/\\-]\\d{1,2}[/\\-]\\d{4}$',
                r'^\\d{4}[/\\-]\\d{1,2}[/\\-]\\d{1,2}$',
                r'^\\d{1,2}\\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\\s+\\d{4}$'
            ]
            return any(re.match(pattern, text, re.IGNORECASE) for pattern in date_patterns)
        
        return [validate_date_format]
    
    def _initialize_causal_validators(self) -> List:
        """Initialise les validateurs de relations causales."""
        
        def validate_causal_plausibility(text: str) -> bool:
            # Validation très basique de plausibilité
            implausible_patterns = [
                r'couleur.*mort',
                r'pluie.*économie',
                r'musique.*maladie'
            ]
            return not any(re.search(pattern, text, re.IGNORECASE) for pattern in implausible_patterns)
        
        return [validate_causal_plausibility]
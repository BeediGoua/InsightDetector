"""
Analyseur de cohérence interne pour détecter contradictions et incohérences factuelles.
"""

import re
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class InternalConsistencyAnalyzer:
    """
    Analyseur de cohérence interne d'un résumé.
    
    Détecte les contradictions internes, incohérences temporelles,
    et violations de la logique factuelle au sein du même texte.
    """
    
    def __init__(self):
        """Initialise l'analyseur de cohérence interne."""
        
        # Patterns pour détecter des éléments factuels
        self.factual_patterns = {
            'dates': [
                r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\b',
                r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b',
                r'\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b'
            ],
            'numbers': [
                r'\b\d+([.,]\d+)?\s*(millions?|milliards?|%|pourcent)\b',
                r'\b\d+([.,]\d+)?\s*(euros?|dollars?|€|\$)\b'
            ],
            'entities': [
                r'\b[A-ZÀ-Ÿ][a-zà-ÿ]+\s+[A-ZÀ-Ÿ][a-zà-ÿ]+\b',  # Noms de personnes
                r'\b[A-ZÀ-Ÿ]{2,}\b'  # Acronymes
            ],
            'temporal_markers': [
                r'\b(avant|après|puis|ensuite|pendant|durant|lors de)\b',
                r'\b(hier|aujourd\'hui|demain)\b',
                r'\b(cette\s+année|l\'année\s+dernière|l\'année\s+prochaine)\b'
            ]
        }
        
        # Compilation des patterns pour performance
        self.compiled_patterns = {}
        for category, patterns in self.factual_patterns.items():
            self.compiled_patterns[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        
        # Mots de négation pour détecter contradictions
        self.negation_words = {
            'ne', 'pas', 'non', 'aucun', 'aucune', 'jamais', 'nullement',
            'point', 'guère', 'plus', 'ni', 'sans'
        }
        
        # Mots de certitude vs incertitude
        self.certainty_markers = {
            'certain': ['certainement', 'définitivement', 'assurément', 'indubitablement'],
            'uncertain': ['peut-être', 'probablement', 'possiblement', 'vraisemblablement', 'semble']
        }
    
    def validate(self, summary_data: Dict) -> Dict:
        """
        Analyse la cohérence interne d'un résumé.
        
        Args:
            summary_data: Données du résumé enrichi
            
        Returns:
            Dict: Résultat de l'analyse de cohérence
        """
        text = summary_data.get('text', '')
        
        # Protection contre les entrées non-string
        if not isinstance(text, str):
            logger.warning(f"Texte non-string reçu: {type(text)}, conversion forcée")
            text = str(text) if text is not None else ""
        
        # Extraction des éléments factuels
        factual_elements = self._extract_factual_elements(text)
        
        # Analyse des contradictions
        contradictions = self._detect_contradictions(text, factual_elements)
        
        # Analyse de la cohérence temporelle
        temporal_consistency = self._analyze_temporal_consistency(text, factual_elements)
        
        # Analyse de la cohérence des entités
        entity_consistency = self._analyze_entity_consistency(text, factual_elements)
        
        # Analyse des marqueurs de certitude
        certainty_consistency = self._analyze_certainty_consistency(text)
        
        # Calcul du score de cohérence interne
        consistency_score = self._calculate_consistency_score(
            contradictions, temporal_consistency, entity_consistency, certainty_consistency
        )
        
        # Collecte des éléments flagués
        flagged_elements = []
        if contradictions['contradictions_found']:
            flagged_elements.extend([f"Contradiction: {c}" for c in contradictions['contradiction_details']])
        if not temporal_consistency['is_consistent']:
            flagged_elements.append(f"Incohérence temporelle: {temporal_consistency['issue_description']}")
        if not entity_consistency['is_consistent']:
            flagged_elements.append(f"Incohérence entités: {entity_consistency['issue_description']}")
        if not certainty_consistency['is_consistent']:
            flagged_elements.append(f"Incohérence certitude: {certainty_consistency['issue_description']}")
        
        return {
            'score': consistency_score,
            'flagged_elements': flagged_elements,
            'analysis_details': {
                'contradictions': contradictions,
                'temporal_consistency': temporal_consistency,
                'entity_consistency': entity_consistency,
                'certainty_consistency': certainty_consistency,
                'factual_elements_count': sum(len(elements) for elements in factual_elements.values())
            },
            'confidence_level': self._calculate_confidence_level(consistency_score, len(flagged_elements))
        }
    
    def _extract_factual_elements(self, text: str) -> Dict[str, List[str]]:
        """Extrait les éléments factuels du texte."""
        
        # Protection contre les entrées non-string
        if not isinstance(text, str):
            logger.warning(f"Texte non-string reçu: {type(text)}, conversion forcée")
            text = str(text) if text is not None else ""
        
        factual_elements = defaultdict(list)
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                try:
                    matches = pattern.findall(text)
                    # Conversion des tuples en strings si nécessaire
                    if matches and isinstance(matches[0], tuple):
                        matches = [' '.join(match) if isinstance(match, tuple) else match for match in matches]
                    factual_elements[category].extend(matches)
                except Exception as e:
                    logger.warning(f"Erreur extraction {category}: {e}")
                    continue
        
        return dict(factual_elements)
    
    def _detect_contradictions(self, text: str, factual_elements: Dict) -> Dict:
        """Détecte les contradictions internes dans le texte."""
        
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        contradictions = []
        
        # Recherche de contradictions entre phrases
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                contradiction = self._check_sentence_contradiction(sentence1, sentence2)
                if contradiction:
                    contradictions.append(contradiction)
        
        # Détection de contradictions numériques
        numerical_contradictions = self._detect_numerical_contradictions(factual_elements.get('numbers', []))
        contradictions.extend(numerical_contradictions)
        
        return {
            'contradictions_found': len(contradictions) > 0,
            'contradiction_count': len(contradictions),
            'contradiction_details': contradictions[:5]  # Limite à 5 pour lisibilité
        }
    
    def _check_sentence_contradiction(self, sentence1: str, sentence2: str) -> str:
        """Vérifie si deux phrases se contredisent avec logique équilibrée."""
        
        # Filtrer les phrases trop courtes
        if len(sentence1.strip()) < 15 or len(sentence2.strip()) < 15:
            return None
            
        # Éviter les fragments de navigation web
        web_fragments = ['partager', 'facebook', 'twitter', 'e-mail', 's\'abonner', 
                        'lecture :', 'mis à jour', 'cliquez', 'voir plus']
        s1_lower = sentence1.lower()
        s2_lower = sentence2.lower()
        
        if any(fragment in s1_lower or fragment in s2_lower for fragment in web_fragments):
            return None
        
        # Détection de contradictions sémantiques directes
        contradiction_found = self._detect_semantic_contradiction(s1_lower, s2_lower)
        if contradiction_found:
            return f"Contradiction potentielle entre: '{sentence1[:50]}...' et '{sentence2[:50]}...'"
        
        # Détection de négations opposées (logique originale mais assouplie)
        s1_words = set(sentence1.lower().split())
        s2_words = set(sentence2.lower().split())
        
        negation_words = {'ne', 'pas', 'non', 'jamais', 'plus', 'rien', 'aucun', 'aucune'}
        s1_has_negation = bool(s1_words & negation_words)
        s2_has_negation = bool(s2_words & negation_words)
        
        if s1_has_negation != s2_has_negation:
            common_words = s1_words & s2_words
            stop_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'à', 'dans', 
                         'sur', 'pour', 'avec', 'par', 'un', 'une', 'ce', 'cette', 'ces',
                         'il', 'elle', 'ils', 'elles', 'qui', 'que', 'dont', 'où',
                         'est', 'sont', 'a', 'ont', 'fait', 'faire', 'dit', 'dire'}
            common_meaningful = common_words - stop_words
            
            # ✅ CORRECTION : Critère plus sévère pour éviter faux positifs
            if len(common_meaningful) >= 3 and len(common_meaningful) <= 8:  # Entre 3 et 8 mots
                # ✅ Vérifier que ce ne sont pas des fragments de navigation
                meaningful_text = ' '.join(common_meaningful).lower()
                if not any(web in meaningful_text for web in web_fragments):
                    return f"Contradiction potentielle entre: '{sentence1[:50]}...' et '{sentence2[:50]}...'"
        
        return None
    
    def _detect_semantic_contradiction(self, sent1: str, sent2: str) -> bool:
        """
        Détecte les contradictions sémantiques directes.
        """
        # Paires de mots contradictoires
        contradiction_pairs = [
            ('gagné', 'perdu'), ('gagne', 'perd'), ('gain', 'perte'),
            ('réussi', 'échoué'), ('succès', 'échec'), ('réussit', 'échoue'),
            ('terminé', 'inachevé'), ('fini', 'commencé'), ('abouti', 'abandonné'),
            ('augmenté', 'diminué'), ('hausse', 'baisse'), ('monte', 'descend'),
            ('oui', 'non'), ('vrai', 'faux'), ('possible', 'impossible'),
            ('présent', 'absent'), ('arrivé', 'parti'), ('ouvert', 'fermé'),
            ('accepté', 'refusé'), ('approuvé', 'rejeté'), ('validé', 'annulé')
        ]
        
        # Chercher des paires contradictoires
        for word1, word2 in contradiction_pairs:
            if (word1 in sent1 and word2 in sent2) or (word2 in sent1 and word1 in sent2):
                return True
        
        # Vérifier les chiffres contradictoires sur même entité
        import re
        numbers1 = re.findall(r'\d+(?:[.,]\d+)?', sent1)
        numbers2 = re.findall(r'\d+(?:[.,]\d+)?', sent2)
        
        if numbers1 and numbers2:
            # Si différents chiffres sur contexte similaire
            words1 = set(sent1.split())
            words2 = set(sent2.split())
            common_context = words1 & words2
            
            # Si contexte similaire mais chiffres différents
            if len(common_context) >= 3 and numbers1 != numbers2:
                return True
        
        return False
    
    def _detect_numerical_contradictions(self, numbers: List[str]) -> List[str]:
        """Détecte les contradictions dans les données numériques."""
        
        contradictions = []
        parsed_numbers = []
        
        # Parser les nombres
        for num_str in numbers:
            try:
                # Extraire le nombre et l'unité
                # Protection contre les types non-string
                num_text = str(num_str) if not isinstance(num_str, str) else num_str
                match = re.search(r'(\d+(?:[.,]\d+)?)\s*(.+)', num_text)
                if match:
                    value = float(match.group(1).replace(',', '.'))
                    unit = match.group(2).lower().strip()
                    parsed_numbers.append((value, unit, num_text))
            except ValueError:
                continue
        
        # Recherche de contradictions
        for i, (val1, unit1, str1) in enumerate(parsed_numbers):
            for val2, unit2, str2 in parsed_numbers[i+1:]:
                if unit1 == unit2:  # Même unité
                    # Vérifier écarts importants (peut indiquer erreur)
                    if val1 > 0 and val2 > 0:
                        ratio = max(val1, val2) / min(val1, val2)
                        if ratio > 10:  # Écart de plus de 10x
                            contradictions.append(f"Écart numérique important: {str1} vs {str2}")
                
                # Vérifier cohérence des ordres de grandeur
                if ('million' in unit1 and 'milliard' in unit2) or ('milliard' in unit1 and 'million' in unit2):
                    if abs(val1 - val2) < 10:  # Valeurs proches mais unités différentes
                        contradictions.append(f"Incohérence d'unités: {str1} vs {str2}")
        
        return contradictions
    
    def _analyze_temporal_consistency(self, text: str, factual_elements: Dict) -> Dict:
        """Analyse la cohérence temporelle du texte."""
        
        dates = factual_elements.get('dates', [])
        temporal_markers = []
        
        # Extraire marqueurs temporels
        for pattern in self.compiled_patterns['temporal_markers']:
            temporal_markers.extend(pattern.findall(text))
        
        # Analyse basique de cohérence temporelle
        is_consistent = True
        issue_description = ""
        
        # Vérifier cohérence des dates
        if len(dates) > 1:
            parsed_dates = []
            for date_str in dates:
                # Tentative de parsing basique des années
                year_match = re.search(r'\b(\d{4})\b', date_str)
                if year_match:
                    parsed_dates.append(int(year_match.group(1)))
            
            if len(parsed_dates) > 1:
                # Vérifier ordre chronologique vs marqueurs temporels
                date_order = sorted(parsed_dates)
                # Si les dates ne sont pas dans l'ordre et qu'il y a des marqueurs temporels
                if parsed_dates != date_order and temporal_markers:
                    is_consistent = False
                    issue_description = f"Ordre chronologique incohérent: {dates[:3]}"
        
        # Vérifier cohérence marqueurs temporels
        has_past_markers = any(marker in ['avant', 'hier', 'dernière'] for marker in temporal_markers)
        has_future_markers = any(marker in ['après', 'demain', 'prochaine'] for marker in temporal_markers)
        
        if has_past_markers and has_future_markers and len(temporal_markers) < 5:
            # Mélange passé/futur dans texte court peut être incohérent
            is_consistent = False
            issue_description = "Mélange de marqueurs temporels passé/futur"
        
        return {
            'is_consistent': is_consistent,
            'issue_description': issue_description,
            'dates_found': len(dates),
            'temporal_markers_found': len(temporal_markers)
        }
    
    def _analyze_entity_consistency(self, text: str, factual_elements: Dict) -> Dict:
        """Analyse la cohérence des entités avec détection sémantique intelligente."""
        
        entities = factual_elements.get('entities', [])
        is_consistent = True
        issue_description = ""
        
        # Grouper les entités par similarité sémantique
        entity_groups = self._group_entities_semantically(entities)
        
        # Vérifier cohérence dans chaque groupe
        for group_key, variations in entity_groups.items():
            if len(variations) > 1:
                unique_variations = list(set(variations))
                if len(unique_variations) > 1:
                    # Analyse sémantique avancée
                    suspicious_pairs = self._find_suspicious_entity_pairs(unique_variations)
                    if suspicious_pairs:
                        is_consistent = False
                        var1, var2 = suspicious_pairs[0]
                        issue_description = f"Variations d'entité suspectes: {var1} vs {var2}"
                        break
        
        return {
            'is_consistent': is_consistent,
            'issue_description': issue_description,
            'entities_found': len(entities),
            'unique_entities': len(set(entities)),
            'semantic_groups': len(entity_groups)
        }
    
    def _group_entities_semantically(self, entities: List[str]) -> Dict[str, List[str]]:
        """
        Groupe les entités par similarité sémantique plutôt que syntaxique.
        """
        entity_groups = defaultdict(list)
        
        for entity in entities:
            if not entity or len(entity.strip()) < 2:
                continue
                
            # Normalisation sémantique
            normalized = self._normalize_entity_semantically(entity)
            entity_groups[normalized].append(entity)
            
        return dict(entity_groups)
    
    def _normalize_entity_semantically(self, entity: str) -> str:
        """
        Normalise une entité pour regroupement sémantique.
        """
        # Nettoyage basique
        clean = entity.lower().strip()
        
        # Supprimer articles et prépositions communes
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'en', 'dans', 'sur', 'par'}
        words = [w for w in clean.split() if w not in stop_words]
        
        # Garder les mots significatifs (noms propres, acronymes, nombres)
        significant_words = []
        for word in words:
            if word.isupper() or (word and word[0].isupper()) or word.isdigit():
                significant_words.append(word)
            elif len(word) > 3:  # Mots longs probablement significatifs
                significant_words.append(word)
        
        # Retourner clé normalisée
        return ' '.join(significant_words[:3])  # Max 3 mots pour la clé
    
    def _find_suspicious_entity_pairs(self, variations: List[str]) -> List[Tuple[str, str]]:
        """
        Trouve les paires d'entités vraiment suspectes (pas les variations normales).
        """
        suspicious_pairs = []
        
        for i, var1 in enumerate(variations):
            for var2 in variations[i+1:]:
                if self._is_suspicious_entity_variation(var1, var2):
                    suspicious_pairs.append((var1, var2))
                    if len(suspicious_pairs) >= 3:  # Limite pour performance
                        break
            if len(suspicious_pairs) >= 3:
                break
                
        return suspicious_pairs
    
    def _analyze_certainty_consistency(self, text: str) -> Dict:
        """Analyse la cohérence des marqueurs de certitude."""
        
        # Compter marqueurs de certitude vs incertitude
        certain_count = 0
        uncertain_count = 0
        
        text_lower = text.lower()
        
        for marker in self.certainty_markers['certain']:
            certain_count += text_lower.count(marker)
        
        for marker in self.certainty_markers['uncertain']:
            uncertain_count += text_lower.count(marker)
        
        is_consistent = True
        issue_description = ""
        
        # Analyser cohérence
        if certain_count > 0 and uncertain_count > 0:
            ratio = certain_count / max(1, uncertain_count)
            if ratio > 3:  # Beaucoup plus de certitude que d'incertitude
                is_consistent = False
                issue_description = f"Excès de certitude vs nuances ({certain_count} vs {uncertain_count})"
            elif uncertain_count > certain_count * 2 and certain_count > 0:
                # Beaucoup d'incertitude mais aussi certitude -> potentiellement incohérent
                is_consistent = False
                issue_description = f"Mélange incohérent certitude/incertitude"
        
        return {
            'is_consistent': is_consistent,
            'issue_description': issue_description,
            'certain_markers': certain_count,
            'uncertain_markers': uncertain_count
        }
    
    def _calculate_consistency_score(self, contradictions: Dict, temporal_consistency: Dict,
                                   entity_consistency: Dict, certainty_consistency: Dict) -> float:
        """Calcule le score de cohérence interne."""
        
        base_score = 1.0
        
        # Pénalités pour chaque type d'incohérence
        if contradictions['contradictions_found']:
            contradiction_penalty = min(0.5, contradictions['contradiction_count'] * 0.2)
            base_score -= contradiction_penalty
        
        if not temporal_consistency['is_consistent']:
            base_score -= 0.3
        
        if not entity_consistency['is_consistent']:
            base_score -= 0.2
        
        if not certainty_consistency['is_consistent']:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_confidence_level(self, consistency_score: float, num_flagged: int) -> str:
        """Calcule le niveau de confiance de l'analyse."""
        
        if consistency_score > 0.8 and num_flagged == 0:
            return 'high'
        elif consistency_score > 0.6 and num_flagged <= 2:
            return 'medium'
        elif consistency_score > 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def _is_suspicious_entity_variation(self, var1: str, var2: str) -> bool:
        """
        Détermine si deux variations d'entité sont réellement suspectes.
        Version corrigée avec logique grammaticale française.
        """
        var1_clean = var1.lower().strip()
        var2_clean = var2.lower().strip()
        
        # Cas évidents non-suspects
        if var1_clean == var2_clean:
            return False
            
        # Constructions grammaticales françaises normales
        if self._is_normal_french_variation(var1_clean, var2_clean):
            return False
            
        # Vérifier différences de chiffres/dates (vraiment suspect)
        import re
        nums1 = re.findall(r'\d+', var1_clean)
        nums2 = re.findall(r'\d+', var2_clean)
        if nums1 != nums2 and (nums1 or nums2):
            return True
        
        # Orthographes très similaires mais différentes (erreurs possibles)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, var1_clean, var2_clean).ratio()
        
        # Suspect uniquement si 80-95% de similarité ET assez long
        if 0.80 <= similarity <= 0.95 and len(var1_clean) > 5 and len(var2_clean) > 5:
            # Vérifier que ce ne sont pas des variations de déclinaisons
            if not self._is_declension_variation(var1_clean, var2_clean):
                return True
            
        return False
    
    def _is_normal_french_variation(self, var1: str, var2: str) -> bool:
        """
        ✅ CORRECTION CRITIQUE : Vérification améliorée des variations normales.
        """
        # Normaliser en supprimant les mots fonctionnels étendus
        def normalize_phrase(phrase):
            words = phrase.split()
            functional_words = {
                # Articles et déterminants
                'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'l',
                # Prépositions
                'dans', 'sur', 'avec', 'par', 'pour', 'en', 'à', 'vers', 'chez', 'sans',
                # Conjonctions
                'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car',
                # Négations
                'ne', 'pas', 'plus', 'jamais', 'rien', 'personne',
                # ✅ AJOUT : Mots fréquents qui créent des variations artificielles
                'qui', 'que', 'dont', 'où', 'se', 's', 'y', 'il', 'elle', 'on',
                'ce', 'cette', 'ces', 'cet', 'sa', 'son', 'ses', 'leur', 'leurs',
                'au', 'aux', 'depuis', 'pendant', 'avant', 'après'
            }
            meaningful_words = [w for w in words if w.lower() not in functional_words and len(w) > 1]
            return ' '.join(meaningful_words)
        
        norm1 = normalize_phrase(var1)
        norm2 = normalize_phrase(var2)
        
        # Si les mots significatifs sont identiques, c'est une variation normale
        if norm1 == norm2 and norm1:  # et pas vide
            return True
        
        # ✅ Vérifications étendues pour variations normales
        # Si une phrase est contenue dans l'autre (expansion)
        if var1 in var2 or var2 in var1:
            return True
            
        # ✅ NOUVEAU : Vérifier les variations de longueur similaire
        if abs(len(var1) - len(var2)) <= 3 and len(var1) > 5:  # Différence mineure
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, var1.lower(), var2.lower()).ratio()
            if similarity > 0.8:  # Très similaires
                return True
        
        # ✅ NOUVEAU : Cas spécifiques du français
        # Pluriels/singuliers
        if (var1.endswith('s') and var1[:-1] == var2) or (var2.endswith('s') and var2[:-1] == var1):
            return True
            
        return False
    
    def _is_declension_variation(self, var1: str, var2: str) -> bool:
        """
        Vérifie si c'est une variation de déclinaison normale.
        """
        # Terminaisons communes de déclinaison
        declension_suffixes = ['s', 'es', 'ent', 'er', 'ir', 'e', 'tion', 'sion']
        
        for suffix in declension_suffixes:
            if var1.endswith(suffix) and var2 == var1[:-len(suffix)]:
                return True
            if var2.endswith(suffix) and var1 == var2[:-len(suffix)]:
                return True
                
        return False
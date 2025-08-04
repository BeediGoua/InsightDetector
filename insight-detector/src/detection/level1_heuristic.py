"""
Niveau 1 : Détection heuristique rapide (<100ms)

Module de détection heuristique pour identifier les hallucinations basées sur :
- Incohérences temporelles (anachronismes, contradictions)
- Validation des entités (NER + cross-check externe)
- Relations causales suspectes

Basé sur l'analyse de 319 résumés validés par le Niveau 0.
"""

import re
import datetime
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
import logging
import time
import spacy
import requests
from collections import defaultdict, Counter
import unicodedata

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patterns temporels français
TEMPORAL_PATTERNS = {
    'dates_absolues': [
        r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b',  # DD/MM/YYYY
        r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b',  # YYYY/MM/DD
        r'\b(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})\b',
    ],
    'dates_relatives': [
        r'\b(hier|aujourd\'?hui|demain)\b',
        r'\b(cette\s+semaine|la\s+semaine\s+dernière|la\s+semaine\s+prochaine)\b',
        r'\b(ce\s+mois|le\s+mois\s+dernier|le\s+mois\s+prochain)\b',
        r'\b(cette\s+année|l\'?année\s+dernière|l\'?année\s+prochaine)\b',
        r'\bil\s+y\s+a\s+(\d+)\s+(jours?|semaines?|mois|années?)\b',
        r'\bdans\s+(\d+)\s+(jours?|semaines?|mois|années?)\b',
    ],
    'evenements_historiques': [
        r'\b(révolution\s+française|première\s+guerre\s+mondiale|seconde\s+guerre\s+mondiale)\b',
        r'\b(renaissance|moyen\s+âge|antiquité|époque\s+moderne)\b',
        r'\b(covid-19|coronavirus|pandémie)\b',
    ]
}

# Entités et personnalités connues pour validation
KNOWN_ENTITIES = {
    'presidents_france': {
        'emmanuel macron': (2017, None),
        'françois hollande': (2012, 2017),
        'nicolas sarkozy': (2007, 2012),
        'jacques chirac': (1995, 2007),
        'françois mitterrand': (1981, 1995),
        'valéry giscard d\'estaing': (1974, 1981, 2020),  # décédé en 2020
    },
    'technologies': {
        'smartphone': (2007, None),  # iPhone lancé en 2007
        'internet': (1990, None),    # Grand public ~1990
        'télévision': (1950, None),
        'radio': (1920, None),
        'automobile': (1885, None),
    }
}

# Relations causales improbables
IMPLAUSIBLE_CAUSALITIES = [
    (r'\bpluie\b', r'\b(licenciements?|faillite|économie)\b'),
    (r'\bsoleil\b', r'\b(politique|élections?)\b'),
    (r'\b(couleur|musique)\b', r'\b(maladie|décès)\b'),
    (r'\b(météo|temps)\b', r'\b(bourse|actions)\b'),
    (r'\b(sport|football)\b', r'\b(sciences?|mathématiques?)\b'),
]


@dataclass
class HeuristicResult:
    """Résultat enrichi de l'analyse heuristique pour alimenter les niveaux suivants."""
    # Évaluation globale (pour compatibilité)
    is_valid: bool
    confidence_score: float  # 0-1
    risk_level: str  # 'low', 'medium', 'high'
    processing_time_ms: float
    
    # Analyses détaillées pour Niveau 2+
    detected_issues: List[str]
    temporal_anomalies: List[Dict]
    entity_issues: List[Dict]
    causal_anomalies: List[Dict]
    
    # NOUVEAUX : Éléments utiles pour la suite
    statistical_profile: Dict  # Profil statistique du texte
    entity_extraction: Dict    # Entités extraites avec confiance
    complexity_metrics: Dict   # Métriques de complexité textuelle
    quality_indicators: Dict   # Indicateurs de qualité pour prioritisation
    fact_check_candidates: List[Dict]  # Éléments prioritaires pour fact-checking
    validation_hints: Dict     # Indices pour validation externe
    
    def get_priority_score(self) -> float:
        """Calcule un score de priorité pour le traitement Niveau 2."""
        base_priority = 1.0 - self.confidence_score
        
        # Bonus selon le type de problèmes détectés
        if self.entity_issues:
            base_priority += 0.2
        if self.temporal_anomalies:
            base_priority += 0.3
        if self.causal_anomalies:
            base_priority += 0.1
            
        # Bonus selon la qualité des métriques existantes
        quality_bonus = self.quality_indicators.get('needs_fact_check', 0) * 0.2
        
        return min(1.0, base_priority + quality_bonus)
    
    def get_fact_check_targets(self) -> List[str]:
        """Retourne les éléments prioritaires pour le fact-checking."""
        targets = []
        
        # Entités suspectes
        for entity in self.entity_issues:
            if entity.get('type') in ['PERSON', 'ORG', 'GPE']:
                targets.append(f"Entity: {entity.get('text', '')}")
        
        # Éléments temporels
        for anomaly in self.temporal_anomalies:
            targets.append(f"Temporal: {anomaly.get('description', '')}")
            
        # Relations causales
        for causal in self.causal_anomalies:
            targets.append(f"Causal: {causal.get('description', '')}")
            
        return targets[:5]  # Top 5 priorités


class Level1HeuristicDetector:
    """
    Détecteur heuristique de niveau 1 pour les hallucinations.
    
    Analyse rapide (<100ms) basée sur des règles heuristiques pour détecter :
    - Les incohérences temporelles
    - Les entités invalides ou suspectes
    - Les relations causales improbables
    """
    
    def __init__(self, 
                 spacy_model: str = "fr_core_news_sm",
                 use_external_validation: bool = False,  # CORRIGÉ - Désactivé par défaut pour performance
                 wikidata_timeout: int = 2,
                 current_year: Optional[int] = None,
                 sensitivity_mode: str = "balanced"):
        """
        Initialise le détecteur heuristique.
        
        Args:
            spacy_model: Modèle spaCy pour l'analyse NER
            use_external_validation: Activer la validation externe (Wikidata)
            wikidata_timeout: Timeout pour les requêtes Wikidata (secondes)
            current_year: Année courante (None = année actuelle)
        """
        # Chargement du modèle spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Modèle spaCy {spacy_model} chargé avec succès")
        except OSError:
            logger.warning(f"Modèle {spacy_model} non trouvé, utilisation du modèle de base")
            self.nlp = spacy.load("fr_core_news_sm")
        
        self.use_external_validation = use_external_validation
        self.wikidata_timeout = wikidata_timeout
        self.current_year = current_year or datetime.datetime.now().year
        self.sensitivity_mode = sensitivity_mode  # "strict", "balanced", "permissive"
        
        # Configuration des seuils selon le mode (CORRIGÉ - plus permissif pour enrichissement)
        if sensitivity_mode == "strict":
            self.confidence_threshold = 0.1
            self.entity_validation_rate = 0.3  # 30% des entités validées
        elif sensitivity_mode == "balanced":
            self.confidence_threshold = 0.05  # Plus permissif pour enrichissement
            self.entity_validation_rate = 0.05  # Validation minimale pour performance
        else:  # permissive
            self.confidence_threshold = 0.02
            self.entity_validation_rate = 0.02  # Validation très réduite
        
        # Compilation des patterns pour performance
        self.temporal_regex = {}
        for category, patterns in TEMPORAL_PATTERNS.items():
            compiled_patterns = []
            for pattern in patterns:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            self.temporal_regex[category] = compiled_patterns
        
        # Patterns de causalité
        self.causality_patterns = []
        for cause_pattern, effect_pattern in IMPLAUSIBLE_CAUSALITIES:
            self.causality_patterns.append((
                re.compile(cause_pattern, re.IGNORECASE),
                re.compile(effect_pattern, re.IGNORECASE)
            ))
        
        # Cache pour les validations externes
        self.entity_cache = {}
        
    def detect_hallucinations(self, text: str, summary_id: Optional[str] = None, 
                             coherence_score: Optional[float] = None,
                             factuality_score: Optional[float] = None,
                             quality_grade: Optional[str] = None) -> HeuristicResult:
        """
        Analyse heuristique complète d'un résumé.
        
        Args:
            text: Texte du résumé à analyser
            summary_id: Identifiant optionnel pour logging
            
        Returns:
            HeuristicResult: Résultat détaillé de l'analyse
        """
        start_time = time.time()
        detected_issues = []
        
        # Stocker le grade et texte pour le calcul de confiance et enrichissement
        self._current_quality_grade = quality_grade
        self._current_text = text
        
        # 1. Analyse statistique agressive (nouveau)
        statistical_issues = self._analyze_statistical_anomalies(text)
        if statistical_issues:
            detected_issues.extend([f"Anomalie statistique: {s['description']}" for s in statistical_issues])
        
        # 2. Analyse de complexité syntaxique (nouveau)
        syntactic_issues = self._analyze_syntactic_complexity(text)
        if syntactic_issues:
            detected_issues.extend([f"Complexité suspecte: {s['description']}" for s in syntactic_issues])
        
        # 3. Détection répétitions agressives (nouveau)
        repetition_issues = self._detect_aggressive_repetitions(text)
        if repetition_issues:
            detected_issues.extend([f"Répétition problématique: {r['description']}" for r in repetition_issues])
        
        # 4. Analyse densité entités (nouveau)
        entity_density_issues = self._analyze_entity_density(text)
        if entity_density_issues:
            detected_issues.extend([f"Densité entités suspecte: {e['description']}" for e in entity_density_issues])
        
        # 5. Détection des incohérences temporelles (existant)
        temporal_anomalies = self._detect_temporal_anomalies(text)
        if temporal_anomalies:
            detected_issues.extend([f"Incohérence temporelle: {a['description']}" for a in temporal_anomalies])
        
        # 6. Validation des entités (existant)
        entity_issues = self._validate_entities(text)
        if entity_issues:
            detected_issues.extend([f"Entité suspecte: {e['description']}" for e in entity_issues])
        
        # 7. Détection des relations causales suspectes (existant)
        causal_anomalies = self._detect_causal_anomalies(text)
        if causal_anomalies:
            detected_issues.extend([f"Relation causale suspecte: {c['description']}" for c in causal_anomalies])
        
        # 8. Intégration des métriques existantes (nouveau)
        metrics_issues = self._analyze_existing_metrics(coherence_score, factuality_score, quality_grade)
        if metrics_issues:
            detected_issues.extend([f"Métrique suspecte: {m['description']}" for m in metrics_issues])
        
        # Calcul du score de confiance et niveau de risque (avec nouvelles anomalies)
        confidence_score, risk_level = self._calculate_confidence(
            temporal_anomalies, entity_issues, causal_anomalies,
            statistical_issues, syntactic_issues, repetition_issues, entity_density_issues, metrics_issues
        )
        
        # Décision finale (seuil adaptatif selon le mode)
        is_valid = confidence_score > self.confidence_threshold and risk_level != 'high'
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Logging pour cas problématiques
        if not is_valid and summary_id:
            logger.warning(f"Résumé {summary_id} suspect - Score: {confidence_score:.3f}, "
                         f"Risque: {risk_level}, Issues: {len(detected_issues)}")
        
        # Génération des nouvelles informations utiles pour les niveaux suivants
        statistical_profile = self._generate_statistical_profile(text, statistical_issues)
        entity_extraction = self._extract_entities_with_confidence(text, entity_issues)
        complexity_metrics = self._calculate_complexity_metrics(text, syntactic_issues)
        quality_indicators = self._generate_quality_indicators(coherence_score, factuality_score, quality_grade)
        fact_check_candidates = self._identify_fact_check_candidates(temporal_anomalies, entity_issues, causal_anomalies)
        validation_hints = self._generate_validation_hints(text, entity_issues, temporal_anomalies)
        
        return HeuristicResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            detected_issues=detected_issues,
            temporal_anomalies=temporal_anomalies,
            entity_issues=entity_issues,
            causal_anomalies=causal_anomalies,
            processing_time_ms=processing_time_ms,
            risk_level=risk_level,
            # Nouveaux champs enrichis
            statistical_profile=statistical_profile,
            entity_extraction=entity_extraction,
            complexity_metrics=complexity_metrics,
            quality_indicators=quality_indicators,
            fact_check_candidates=fact_check_candidates,
            validation_hints=validation_hints
        )
    
    def process_batch(self, summaries: List[Dict]) -> Tuple[List[Dict], List[HeuristicResult]]:
        """
        Traite un lot de résumés.
        
        Args:
            summaries: Liste de dictionnaires avec 'text' et optionnellement 'id'
            
        Returns:
            Tuple[valid_summaries, all_results]: Résumés valides et tous les résultats
        """
        valid_summaries = []
        all_results = []
        
        for i, summary in enumerate(summaries):
            text = summary.get('text', '')
            summary_id = summary.get('id', f'summary_{i}')
            
            # Extraire les métriques si disponibles
            coherence_score = summary.get('coherence')
            factuality_score = summary.get('factuality')
            quality_grade = summary.get('quality_grade')
            
            result = self.detect_hallucinations(text, summary_id, coherence_score, factuality_score, quality_grade)
            all_results.append(result)
            
            if result.is_valid:
                valid_summaries.append(summary)
        
        # Statistiques globales
        total_count = len(summaries)
        valid_count = len(valid_summaries)
        rejection_rate = (total_count - valid_count) / total_count * 100
        avg_time = sum(r.processing_time_ms for r in all_results) / total_count
        
        logger.info(f"Niveau 1 - Batch traité: {valid_count}/{total_count} valides "
                   f"({rejection_rate:.1f}% rejetés), temps moyen: {avg_time:.1f}ms")
        
        return valid_summaries, all_results
    
    def _detect_temporal_anomalies(self, text: str) -> List[Dict]:
        """Détecte les incohérences temporelles."""
        anomalies = []
        
        # 1. Extraction des mentions temporelles
        temporal_mentions = self._extract_temporal_mentions(text)
        
        # 2. Vérification des anachronismes
        anachronisms = self._detect_anachronisms(text, temporal_mentions)
        anomalies.extend(anachronisms)
        
        # Nouveaux patterns de détection plus sensibles
        additional_patterns = self._detect_additional_temporal_patterns(text)
        anomalies.extend(additional_patterns)
        
        # 3. Vérification des contradictions temporelles internes
        contradictions = self._detect_temporal_contradictions(temporal_mentions)
        anomalies.extend(contradictions)
        
        # 4. Vérification des dates impossibles
        impossible_dates = self._detect_impossible_dates(temporal_mentions)
        anomalies.extend(impossible_dates)
        
        return anomalies
    
    def _extract_temporal_mentions(self, text: str) -> List[Dict]:
        """Extrait toutes les mentions temporelles du texte."""
        mentions = []
        
        for category, patterns in self.temporal_regex.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    mentions.append({
                        'type': category,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'pattern': pattern.pattern
                    })
        
        return mentions
    
    def _detect_anachronisms(self, text: str, temporal_mentions: List[Dict]) -> List[Dict]:
        """Détecte les anachronismes évidents."""
        anachronisms = []
        
        # Technologies et périodes historiques
        for tech, (start_year, end_year) in KNOWN_ENTITIES['technologies'].items():
            if tech in text.lower():
                for mention in temporal_mentions:
                    if mention['type'] == 'dates_absolues':
                        # Extraction de l'année de la mention
                        year_match = re.search(r'\b(\d{4})\b', mention['text'])
                        if year_match:
                            year = int(year_match.group(1))
                            # Vérification anachronisme
                            if year < start_year:
                                anachronisms.append({
                                    'type': 'anachronisme_technologique',
                                    'description': f"{tech} mentionné en {year} (invention: {start_year})",
                                    'confidence': 0.9,
                                    'text_span': mention['text']
                                })
                            elif end_year and year > end_year:
                                anachronisms.append({
                                    'type': 'anachronisme_technologique',
                                    'description': f"{tech} mentionné en {year} (fin: {end_year})",
                                    'confidence': 0.8,
                                    'text_span': mention['text']
                                })
        
        return anachronisms
    
    def _detect_additional_temporal_patterns(self, text: str) -> List[Dict]:
        """Détecte des patterns temporels suspects supplémentaires."""
        patterns = []
        
        # 1. Dates futures suspectes dans des contextes passés
        future_past_pattern = re.compile(r'(en|depuis|après)\s+(\d{4})\s+.*\s+(sera|aura|devrait)', re.IGNORECASE)
        for match in future_past_pattern.finditer(text):
            year = int(match.group(2))
            if year > self.current_year:
                patterns.append({
                    'type': 'confusion_temporelle',
                    'description': f"Mélange temps futur/passé avec année {year}",
                    'confidence': 0.4,
                    'text_span': match.group()
                })
        
        # 2. Expressions temporelles contradictoires
        contradiction_patterns = [
            (r'hier.*demain', 'hier et demain dans même phrase'),
            (r'passé.*futur.*maintenant', 'confusion temporelle multiple'),
            (r'avant.*après.*simultanément', 'contradiction temporelle'),
        ]
        
        for pattern, description in contradiction_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                patterns.append({
                    'type': 'expression_temporelle_contradictoire',
                    'description': description,
                    'confidence': 0.3,
                    'pattern': pattern
                })
        
        # 3. Dates impossibles avec événements connus
        covid_before_2019 = re.search(r'covid.*201[0-8]', text, re.IGNORECASE)
        if covid_before_2019:
            patterns.append({
                'type': 'anachronisme_covid',
                'description': "COVID-19 mentionné avant 2019",
                'confidence': 0.8,
                'text_span': covid_before_2019.group()
            })
        
        return patterns
    
    def _detect_temporal_contradictions(self, temporal_mentions: List[Dict]) -> List[Dict]:
        """Détecte les contradictions temporelles internes."""
        contradictions = []
        
        # Recherche de contradictions entre dates relatives
        relative_mentions = [m for m in temporal_mentions if m['type'] == 'dates_relatives']
        
        if len(relative_mentions) >= 2:
            # Exemple simple: "hier" et "l'année prochaine" dans le même texte
            has_past = any('hier' in m['text'] or 'dernière' in m['text'] for m in relative_mentions)
            has_future = any('demain' in m['text'] or 'prochaine' in m['text'] for m in relative_mentions)
            
            if has_past and has_future:
                # Vérification plus fine nécessaire
                past_mentions = [m for m in relative_mentions if 'hier' in m['text'] or 'dernière' in m['text']]
                future_mentions = [m for m in relative_mentions if 'demain' in m['text'] or 'prochaine' in m['text']]
                
                if len(past_mentions) > 1 or len(future_mentions) > 1:
                    contradictions.append({
                        'type': 'contradiction_temporelle',
                        'description': f"Références temporelles contradictoires détectées",
                        'confidence': 0.6,
                        'mentions': past_mentions + future_mentions
                    })
        
        return contradictions
    
    def _detect_impossible_dates(self, temporal_mentions: List[Dict]) -> List[Dict]:
        """Détecte les dates impossibles."""
        impossible = []
        
        for mention in temporal_mentions:
            if mention['type'] == 'dates_absolues':
                # Vérification des formats de date
                date_match = re.search(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', mention['text'])
                if date_match:
                    day, month, year = map(int, date_match.groups())
                    
                    # Vérifications de base
                    if month > 12 or month < 1:
                        impossible.append({
                            'type': 'date_impossible',
                            'description': f"Mois invalide: {month}",
                            'confidence': 1.0,
                            'text_span': mention['text']
                        })
                    elif day > 31 or day < 1:
                        impossible.append({
                            'type': 'date_impossible',
                            'description': f"Jour invalide: {day}",
                            'confidence': 1.0,
                            'text_span': mention['text']
                        })
                    elif year > self.current_year + 5:  # Dates futures suspectes (plus strict)
                        impossible.append({
                            'type': 'date_suspecte',
                            'description': f"Date future suspecte: {year}",
                            'confidence': 0.6,
                            'text_span': mention['text']
                        })
                    elif year < 1900:  # Dates anciennes suspectes (plus strict)
                        impossible.append({
                            'type': 'date_suspecte',
                            'description': f"Date ancienne suspecte: {year}",
                            'confidence': 0.4,
                            'text_span': mention['text']
                        })
                    # Nouveaux patterns de détection
                    elif year == self.current_year + 1 and month > datetime.datetime.now().month:
                        impossible.append({
                            'type': 'date_future_proche',
                            'description': f"Date future proche suspecte: {day}/{month}/{year}",
                            'confidence': 0.3,
                            'text_span': mention['text']
                        })
        
        return impossible
    
    def _validate_entities(self, text: str) -> List[Dict]:
        """Valide les entités nommées."""
        issues = []
        
        # Analyse NER avec spaCy
        doc = self.nlp(text)
        
        entity_count = 0
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MISC']:  # Ajout de MISC
                entity_text = ent.text.lower().strip()
                entity_count += 1
                
                # 1. Vérification dans les bases connues
                entity_issue = self._check_known_entities(entity_text, ent.label_)
                if entity_issue:
                    issues.append(entity_issue)
                
                # 2. Validation externe (sélective selon le mode)
                should_validate = (
                    self.use_external_validation and 
                    len(entity_text) > 3 and
                    (entity_count / max(1, len(doc.ents))) <= self.entity_validation_rate
                )
                
                if should_validate:
                    external_issue = self._validate_entity_external(entity_text, ent.label_)
                    if external_issue:
                        issues.append(external_issue)
                
                # 3. Détection de patterns suspects (plus agressive)
                pattern_issue = self._detect_suspicious_entity_patterns(entity_text, ent.label_)
                if pattern_issue:
                    issues.append(pattern_issue)
                
                # 4. Nouveaux patterns de détection
                length_issue = self._check_entity_length_anomalies(entity_text, ent.label_)
                if length_issue:
                    issues.append(length_issue)
                
                # 5. Détection de caractères non-alphabétiques suspects
                char_issue = self._check_entity_character_anomalies(entity_text, ent.label_)
                if char_issue:
                    issues.append(char_issue)
        
        return issues
    
    def _check_known_entities(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """Vérifie une entité contre les bases de données connues."""
        # Vérification des présidents français
        if entity_type == 'PERSON':
            # Normalisation du nom
            normalized_name = self._normalize_name(entity_text)
            
            # Recherche dans les présidents connus
            for president, period in KNOWN_ENTITIES['presidents_france'].items():
                if normalized_name in president or president in normalized_name:
                    return None  # Entité connue et valide
            
            # Détection de variations suspectes
            for president in KNOWN_ENTITIES['presidents_france'].keys():
                similarity = self._name_similarity(normalized_name, president)
                if 0.6 < similarity < 0.9:  # Similitude suspecte mais pas exacte
                    return {
                        'type': 'entite_similaire_suspecte',
                        'description': f"'{entity_text}' similaire à '{president}' ({similarity:.2f})",
                        'confidence': 0.7,
                        'entity': entity_text,
                        'reference': president
                    }
        
        return None
    
    def _validate_entity_external(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """Validation externe via Wikidata (avec cache)."""
        cache_key = f"{entity_text}_{entity_type}"
        
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        try:
            # Requête Wikidata simplifiée
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
                if len(results) > 1 and len(results[1]) > 0:
                    # Entité trouvée dans Wikidata
                    self.entity_cache[cache_key] = None
                    return None
                else:
                    # Entité non trouvée
                    issue = {
                        'type': 'entite_non_verifiee',
                        'description': f"'{entity_text}' non trouvé dans Wikidata",
                        'confidence': 0.4,
                        'entity': entity_text
                    }
                    self.entity_cache[cache_key] = issue
                    return issue
        
        except requests.RequestException:
            # Erreur réseau - pas de pénalité
            self.entity_cache[cache_key] = None
            return None
        
        return None
    
    def _detect_suspicious_entity_patterns(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """Détecte des patterns suspects dans les entités."""
        # Noms de personnes avec des caractères suspects
        if entity_type == 'PERSON':
            # Noms avec des chiffres
            if re.search(r'\d', entity_text):
                return {
                    'type': 'nom_avec_chiffres',
                    'description': f"Nom de personne avec chiffres: '{entity_text}'",
                    'confidence': 0.6,
                    'entity': entity_text
                }
            
            # Noms trop courts ou trop longs
            if len(entity_text.split()) == 1 and len(entity_text) < 3:
                return {
                    'type': 'nom_trop_court',
                    'description': f"Nom très court: '{entity_text}'",
                    'confidence': 0.3,
                    'entity': entity_text
                }
            elif len(entity_text) > 50:
                return {
                    'type': 'nom_trop_long',
                    'description': f"Nom très long: '{entity_text[:30]}...'",
                    'confidence': 0.7,
                    'entity': entity_text
                }
        
        return None
    
    def _check_entity_length_anomalies(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """Détecte les anomalies de longueur dans les entités."""
        if entity_type == 'PERSON':
            # Noms de personnes très courts ou très longs
            if len(entity_text) < 2:
                return {
                    'type': 'nom_trop_court',
                    'description': f"Nom de personne très court: '{entity_text}'",
                    'confidence': 0.4,
                    'entity': entity_text
                }
            elif len(entity_text) > 60:
                return {
                    'type': 'nom_trop_long',
                    'description': f"Nom de personne très long: '{entity_text[:30]}...'",
                    'confidence': 0.6,
                    'entity': entity_text
                }
        
        elif entity_type == 'ORG':
            # Organisations avec des noms suspects
            if len(entity_text) < 2:
                return {
                    'type': 'org_trop_courte',
                    'description': f"Nom d'organisation très court: '{entity_text}'",
                    'confidence': 0.5,
                    'entity': entity_text
                }
        
        return None
    
    def _check_entity_character_anomalies(self, entity_text: str, entity_type: str) -> Optional[Dict]:
        """Détecte les caractères suspects dans les entités."""
        # Caractères non-alphabétiques suspects dans les noms
        if entity_type == 'PERSON':
            # Nombres dans les noms de personnes
            if re.search(r'\d{2,}', entity_text):  # 2+ chiffres consécutifs
                return {
                    'type': 'nom_avec_chiffres_suspects',
                    'description': f"Nom avec chiffres suspects: '{entity_text}'",
                    'confidence': 0.7,
                    'entity': entity_text
                }
            
            # Caractères spéciaux suspects
            if re.search(r'[#@$%&*+=<>{}[\]|\\]', entity_text):
                return {
                    'type': 'nom_caracteres_speciaux',
                    'description': f"Nom avec caractères spéciaux: '{entity_text}'",
                    'confidence': 0.8,
                    'entity': entity_text
                }
        
        # Mots entièrement en majuscules suspects (si pas acronyme)
        if len(entity_text) > 4 and entity_text.isupper() and not re.match(r'^[A-Z]{2,5}$', entity_text):
            return {
                'type': 'entite_majuscules_suspecte',
                'description': f"Entité en majuscules suspecte: '{entity_text}'",
                'confidence': 0.4,
                'entity': entity_text
            }
        
        return None
    
    def _analyze_statistical_anomalies(self, text: str) -> List[Dict]:
        """Analyse statistique agressive du texte."""
        anomalies = []
        
        words = text.split()
        word_count = len(words)
        
        # 1. Longueur suspecte (basé sur analyse des données réelles)
        if word_count < 20:  # Très court
            anomalies.append({
                'type': 'longueur_trop_courte',
                'description': f"Résumé très court: {word_count} mots",
                'confidence': 0.6,
                'value': word_count
            })
        elif word_count > 500:  # Très long
            anomalies.append({
                'type': 'longueur_excessive',
                'description': f"Résumé très long: {word_count} mots",
                'confidence': 0.7,
                'value': word_count
            })
        elif word_count > 400:  # Long
            anomalies.append({
                'type': 'longueur_suspecte',
                'description': f"Résumé potentiellement trop long: {word_count} mots",
                'confidence': 0.4,
                'value': word_count
            })
        
        # 2. Ratio ponctuation/mots suspect
        punct_count = len(re.findall(r'[^\w\s]', text))
        if word_count > 0:
            punct_ratio = punct_count / word_count
            if punct_ratio > 0.15:  # Plus de 15% de ponctuation
                anomalies.append({
                    'type': 'ponctuation_excessive',
                    'description': f"Ratio ponctuation élevé: {punct_ratio:.2f}",
                    'confidence': 0.5,
                    'value': punct_ratio
                })
        
        # 3. Diversité lexicale faible
        unique_words = len(set(word.lower() for word in words if len(word) > 2))
        if word_count > 10:
            lexical_diversity = unique_words / word_count
            if lexical_diversity < 0.4:  # Moins de 40% de mots uniques
                anomalies.append({
                    'type': 'diversite_lexicale_faible',
                    'description': f"Diversité lexicale faible: {lexical_diversity:.2f}",
                    'confidence': 0.4,
                    'value': lexical_diversity
                })
        
        return anomalies
    
    def _analyze_syntactic_complexity(self, text: str) -> List[Dict]:
        """Analyse de la complexité syntaxique."""
        issues = []
        
        # 1. Longueur moyenne des phrases
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            if avg_sentence_length < 5:  # Phrases très courtes
                issues.append({
                    'type': 'phrases_trop_courtes',
                    'description': f"Phrases très courtes: {avg_sentence_length:.1f} mots/phrase",
                    'confidence': 0.5,
                    'value': avg_sentence_length
                })
            elif avg_sentence_length > 30:  # Phrases très longues
                issues.append({
                    'type': 'phrases_trop_longues',
                    'description': f"Phrases très longues: {avg_sentence_length:.1f} mots/phrase",
                    'confidence': 0.6,
                    'value': avg_sentence_length
                })
        
        # 2. Manque de connecteurs logiques
        connectors = ['donc', 'ainsi', 'par conséquent', 'cependant', 'néanmoins', 'toutefois', 
                     'en effet', 'de plus', 'par ailleurs', 'en revanche']
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        word_count = len(text.split())
        
        if word_count > 100 and connector_count == 0:
            issues.append({
                'type': 'manque_connecteurs',
                'description': "Absence de connecteurs logiques dans un texte long",
                'confidence': 0.3,
                'value': connector_count
            })
        
        # 3. Structure répétitive (même début de phrases)
        if len(sentences) > 3:
            sentence_starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
            start_counts = Counter(sentence_starts)
            max_repetition = max(start_counts.values()) if start_counts else 0
            
            if max_repetition > len(sentences) // 3:  # Plus d'1/3 des phrases commencent pareil
                issues.append({
                    'type': 'structure_repetitive',
                    'description': f"Structure répétitive: {max_repetition} phrases commencent de même",
                    'confidence': 0.4,
                    'value': max_repetition
                })
        
        return issues
    
    def _detect_aggressive_repetitions(self, text: str) -> List[Dict]:
        """Détection agressive des répétitions."""
        issues = []
        
        # 1. Répétitions de mots simples
        words = [w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 3]
        word_counts = Counter(words)
        total_words = len(words)
        
        for word, count in word_counts.most_common(10):
            frequency = count / total_words if total_words > 0 else 0
            if frequency > 0.15 and count > 6:  # Plus de 15% et au moins 7 occurrences (CORRIGÉ - moins agressif)
                issues.append({
                    'type': 'mot_trop_frequent',
                    'description': f"Mot '{word}' répété {count} fois ({frequency:.1%})",
                    'confidence': min(0.8, frequency * 10),  # Confiance proportionnelle
                    'word': word,
                    'count': count
                })
        
        # 2. Répétitions de bigrammes
        words_list = text.lower().split()
        if len(words_list) > 1:
            bigrams = [f"{words_list[i]} {words_list[i+1]}" for i in range(len(words_list)-1)]
            bigram_counts = Counter(bigrams)
            
            for bigram, count in bigram_counts.most_common(5):
                if count > 4:  # Bigramme répété plus de 4 fois (CORRIGÉ - moins sensible)
                    issues.append({
                        'type': 'bigramme_repetitif',
                        'description': f"Expression '{bigram}' répétée {count} fois",
                        'confidence': min(0.7, count * 0.2),
                        'bigram': bigram,
                        'count': count
                    })
        
        # 3. Phrases très similaires (distance de Levenshtein simple)
        sentences = [s.strip().lower() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) > 2:
            similar_count = 0
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    # Similarité simple basée sur les mots communs
                    words_i = set(sentences[i].split())
                    words_j = set(sentences[j].split())
                    if words_i and words_j:
                        similarity = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                        if similarity > 0.7:  # 70% de mots communs
                            similar_count += 1
            
            if similar_count > 0:
                issues.append({
                    'type': 'phrases_similaires',
                    'description': f"{similar_count} paires de phrases très similaires",
                    'confidence': min(0.6, similar_count * 0.2),
                    'count': similar_count
                })
        
        return issues
    
    def _analyze_entity_density(self, text: str) -> List[Dict]:
        """Analyse de la densité des entités nommées."""
        issues = []
        
        # Analyse NER avec spaCy
        doc = self.nlp(text)
        entities = [ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MISC']]
        
        words = text.split()
        word_count = len(words)
        entity_count = len(entities)
        
        if word_count > 0:
            entity_density = entity_count / word_count
            
            # 1. Densité d'entités trop faible
            if word_count > 50 and entity_density < 0.02:  # Moins de 2% d'entités dans un texte long
                issues.append({
                    'type': 'densite_entites_faible',
                    'description': f"Densité d'entités faible: {entity_density:.1%} ({entity_count}/{word_count})",
                    'confidence': 0.4,
                    'density': entity_density,
                    'entity_count': entity_count
                })
            
            # 2. Densité d'entités trop élevée
            elif entity_density > 0.15:  # Plus de 15% d'entités
                issues.append({
                    'type': 'densite_entites_elevee',
                    'description': f"Densité d'entités élevée: {entity_density:.1%} ({entity_count}/{word_count})",
                    'confidence': 0.5,
                    'density': entity_density,
                    'entity_count': entity_count
                })
        
        # 3. Types d'entités déséquilibrés
        entity_types = [ent.label_ for ent in entities]
        if entity_types:
            type_counts = Counter(entity_types)
            dominant_type_count = max(type_counts.values())
            
            if len(type_counts) > 1 and dominant_type_count > len(entity_types) * 0.8:
                dominant_type = max(type_counts, key=type_counts.get)
                issues.append({
                    'type': 'entites_desequilibrees',
                    'description': f"Type d'entité dominant: {dominant_type} ({dominant_type_count}/{len(entity_types)})",
                    'confidence': 0.3,
                    'dominant_type': dominant_type,
                    'ratio': dominant_type_count / len(entity_types)
                })
        
        return issues
    
    def _analyze_existing_metrics(self, coherence_score: Optional[float], 
                                 factuality_score: Optional[float], 
                                 quality_grade: Optional[str]) -> List[Dict]:
        """Analyse les métriques existantes pour détecter des problèmes."""
        issues = []
        
        # 1. Score de cohérence suspect (basé sur données réelles)
        if coherence_score is not None:
            if coherence_score < 0.3:  # Cohérence très faible
                issues.append({
                    'type': 'coherence_tres_faible',
                    'description': f"Cohérence très faible: {coherence_score:.3f}",
                    'confidence': 0.8,
                    'value': coherence_score
                })
            elif coherence_score < 0.5:  # Cohérence faible
                issues.append({
                    'type': 'coherence_faible',
                    'description': f"Cohérence faible: {coherence_score:.3f}",
                    'confidence': 0.6,
                    'value': coherence_score
                })
            elif coherence_score < 0.7:  # Cohérence suspecte
                issues.append({
                    'type': 'coherence_suspecte',
                    'description': f"Cohérence en dessous de la moyenne: {coherence_score:.3f}",
                    'confidence': 0.3,
                    'value': coherence_score
                })
        
        # 2. Score de factualité suspect
        if factuality_score is not None:
            if factuality_score < 0.7:  # Factualité faible
                issues.append({
                    'type': 'factualite_faible',
                    'description': f"Factualité faible: {factuality_score:.3f}",
                    'confidence': 0.7,
                    'value': factuality_score
                })
            elif factuality_score < 0.9:  # Factualité suspecte
                issues.append({
                    'type': 'factualite_suspecte',
                    'description': f"Factualité en dessous de la moyenne: {factuality_score:.3f}",
                    'confidence': 0.4,
                    'value': factuality_score
                })
        
        # 3. Grade de qualité problématique
        if quality_grade is not None:
            if quality_grade == 'D':
                issues.append({
                    'type': 'grade_d',
                    'description': f"Grade de qualité D (problématique)",
                    'confidence': 0.9,
                    'grade': quality_grade
                })
            elif quality_grade == 'C':
                issues.append({
                    'type': 'grade_c',
                    'description': f"Grade de qualité C (faible)",
                    'confidence': 0.6,
                    'grade': quality_grade
                })
            elif quality_grade == 'B':
                issues.append({
                    'type': 'grade_b',
                    'description': f"Grade de qualité B (moyen)",
                    'confidence': 0.3,
                    'grade': quality_grade
                })
        
        # 4. Corrélation suspecte entre métriques
        if coherence_score is not None and factuality_score is not None:
            # Si cohérence faible mais factualité élevée (suspect)
            if coherence_score < 0.5 and factuality_score > 0.9:
                issues.append({
                    'type': 'correlation_suspecte',
                    'description': f"Corrélation suspecte: cohérence faible ({coherence_score:.3f}) mais factualité élevée ({factuality_score:.3f})",
                    'confidence': 0.5,
                    'coherence': coherence_score,
                    'factuality': factuality_score
                })
            # Si factualité faible mais cohérence élevée (aussi suspect)
            elif factuality_score < 0.7 and coherence_score > 0.8:
                issues.append({
                    'type': 'correlation_inversee',
                    'description': f"Corrélation inversée: factualité faible ({factuality_score:.3f}) mais cohérence élevée ({coherence_score:.3f})",
                    'confidence': 0.4,
                    'coherence': coherence_score,
                    'factuality': factuality_score
                })
        
        return issues
    
    def _detect_causal_anomalies(self, text: str) -> List[Dict]:
        """Détecte les relations causales suspectes."""
        anomalies = []
        
        for cause_pattern, effect_pattern in self.causality_patterns:
            cause_matches = list(cause_pattern.finditer(text))
            effect_matches = list(effect_pattern.finditer(text))
            
            if cause_matches and effect_matches:
                # Vérification de la proximité dans le texte
                for cause_match in cause_matches:
                    for effect_match in effect_matches:
                        distance = abs(cause_match.start() - effect_match.start())
                        
                        # Si les deux mentions sont proches (<200 caractères)
                        if distance < 200:
                            anomalies.append({
                                'type': 'relation_causale_suspecte',
                                'description': f"Relation improbable: '{cause_match.group()}' → '{effect_match.group()}'",
                                'confidence': 0.6,
                                'cause': cause_match.group(),
                                'effect': effect_match.group(),
                                'distance': distance
                            })
        
        return anomalies
    
    def _calculate_confidence(self, temporal_anomalies: List[Dict], 
                            entity_issues: List[Dict], 
                            causal_anomalies: List[Dict],
                            statistical_issues: List[Dict] = None,
                            syntactic_issues: List[Dict] = None,
                            repetition_issues: List[Dict] = None,
                            entity_density_issues: List[Dict] = None,
                            metrics_issues: List[Dict] = None) -> Tuple[float, str]:
        """Calcule le score de confiance et le niveau de risque."""
        base_confidence = 1.0
        
        # Initialisation des listes vides si None
        statistical_issues = statistical_issues or []
        syntactic_issues = syntactic_issues or []
        repetition_issues = repetition_issues or []
        entity_density_issues = entity_density_issues or []
        metrics_issues = metrics_issues or []
        
        # Pénalités réduites pour privilégier l'enrichissement (CORRIGÉ)
        # Bonus pour grades de qualité élevés
        quality_bonus = 0.0
        if hasattr(self, '_current_quality_grade'):
            if self._current_quality_grade in ['A+', 'A']:
                quality_bonus = 0.4
            elif self._current_quality_grade == 'B+':
                quality_bonus = 0.2
        base_confidence += quality_bonus
        
        # 0. Métriques existantes (impact réduit)
        for issue in metrics_issues:
            penalty = issue.get('confidence', 0.5) * 0.3  # Réduit de 0.7 à 0.3
            base_confidence -= penalty
        
        # 1. Anomalies statistiques (impact réduit)
        for issue in statistical_issues:
            penalty = issue.get('confidence', 0.5) * 0.2  # Réduit de 0.6 à 0.2
            base_confidence -= penalty
        
        # 2. Répétitions (impact réduit)
        for issue in repetition_issues:
            penalty = issue.get('confidence', 0.5) * 0.15  # Réduit de 0.5 à 0.15
            base_confidence -= penalty
        
        # 3. Complexité syntaxique (impact réduit)
        for issue in syntactic_issues:
            penalty = issue.get('confidence', 0.5) * 0.1  # Réduit de 0.4 à 0.1
            base_confidence -= penalty
        
        # 4. Densité d'entités (impact minimal)
        for issue in entity_density_issues:
            penalty = issue.get('confidence', 0.5) * 0.05  # Réduit de 0.3 à 0.05
            base_confidence -= penalty
        
        # 5. Anomalies temporelles (impact réduit)
        for anomaly in temporal_anomalies:
            penalty = anomaly.get('confidence', 0.5) * 0.2  # Réduit de 0.4 à 0.2
            base_confidence -= penalty
        
        # 6. Problèmes d'entités (impact réduit)
        for issue in entity_issues:
            penalty = issue.get('confidence', 0.5) * 0.1  # Réduit de 0.3 à 0.1
            base_confidence -= penalty
        
        # 7. Relations causales (impact minimal)
        for anomaly in causal_anomalies:
            penalty = anomaly.get('confidence', 0.5) * 0.05  # Réduit de 0.2 à 0.05
            base_confidence -= penalty
        
        # Calcul du total d'issues
        total_issues = (len(metrics_issues) + len(statistical_issues) + len(syntactic_issues) + 
                       len(repetition_issues) + len(entity_density_issues) +
                       len(temporal_anomalies) + len(entity_issues) + len(causal_anomalies))
        
        # Bonus/Malus selon le nombre total d'issues (CORRIGÉ - moins pénalisant)
        if total_issues == 0:
            # Pas d'anomalies détectées - bonus léger
            base_confidence += 0.05
        elif total_issues >= 8:  # Augmenté le seuil de 3 à 8
            # Beaucoup d'anomalies - pénalité réduite
            base_confidence -= 0.05  # Réduit de 0.1 à 0.05
        
        # Normalisation
        confidence_score = max(0.0, min(1.0, base_confidence))
        
        # Détermination du niveau de risque (CORRIGÉ - moins agressif)
        if confidence_score < 0.1 or total_issues >= 10:  # Seuils plus permissifs
            risk_level = 'high'
        elif confidence_score < 0.3 or total_issues >= 6:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return confidence_score, risk_level
    
    def _normalize_name(self, name: str) -> str:
        """Normalise un nom pour la comparaison."""
        # Suppression des accents
        name = unicodedata.normalize('NFD', name)
        name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
        
        # Minuscules et suppression de la ponctuation
        name = re.sub(r'[^\w\s]', '', name.lower())
        
        return name.strip()
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calcule la similarité entre deux noms (Jaccard simple)."""
        set1 = set(name1.split())
        set2 = set(name2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_statistical_profile(self, text: str, statistical_issues: List[Dict]) -> Dict:
        """Génère le profil statistique du texte."""
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / max(1, len(text)),
            'lexical_diversity': len(set(words)) / max(1, len(words)),
            'statistical_anomalies_count': len(statistical_issues),
            'has_length_issues': any(issue.get('type') in ['longueur_trop_courte', 'longueur_excessive'] for issue in statistical_issues),
            'has_punctuation_issues': any(issue.get('type') == 'ponctuation_excessive' for issue in statistical_issues)
        }
    
    def _extract_entities_with_confidence(self, text: str, entity_issues: List[Dict]) -> Dict:
        """Extrait les entités avec confiance."""
        doc = self.nlp(text)
        
        return {
            'persons': [{'text': ent.text, 'confidence': 0.5 if any(issue.get('entity') == ent.text for issue in entity_issues) else 1.0} 
                        for ent in doc.ents if ent.label_ == 'PERSON'],
            'organizations': [{'text': ent.text, 'confidence': 0.5 if any(issue.get('entity') == ent.text for issue in entity_issues) else 1.0} 
                             for ent in doc.ents if ent.label_ in ['ORG', 'NORP']],
            'locations': [{'text': ent.text, 'confidence': 0.5 if any(issue.get('entity') == ent.text for issue in entity_issues) else 1.0} 
                          for ent in doc.ents if ent.label_ in ['GPE', 'LOC']],
            'suspicious_entities': len(entity_issues),
            'total_entities': len(doc.ents)
        }
    
    def _calculate_complexity_metrics(self, text: str, syntactic_issues: List[Dict]) -> Dict:
        """Calcule les métriques de complexité."""
        words = text.split()
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        
        return {
            'syntactic_issues_count': len(syntactic_issues),
            'has_length_issues': any(issue.get('type') in ['phrases_trop_longues', 'phrases_trop_courtes'] for issue in syntactic_issues),
            'has_connector_issues': any(issue.get('type') == 'manque_connecteurs' for issue in syntactic_issues),
            'readability_score': min(1.0, (len(words) / max(1, len(sentences))) / 20)
        }
    
    def _generate_quality_indicators(self, coherence_score: Optional[float], factuality_score: Optional[float], quality_grade: Optional[str]) -> Dict:
        """Génère les indicateurs de qualité."""
        needs_fact_check = 0.0
        if coherence_score and coherence_score < 0.3:
            needs_fact_check += 0.5
        if factuality_score and factuality_score < 0.7:
            needs_fact_check += 0.3
        if quality_grade in ['D', 'C']:
            needs_fact_check += 0.4
        
        return {
            'has_coherence_score': coherence_score is not None,
            'has_factuality_score': factuality_score is not None,
            'has_quality_grade': quality_grade is not None,
            'needs_fact_check': min(1.0, needs_fact_check),
            'priority_level': 'high' if quality_grade in ['D', 'C'] else 'medium' if quality_grade == 'B' else 'low'
        }
    
    def _identify_fact_check_candidates(self, temporal_anomalies: List[Dict], entity_issues: List[Dict], causal_anomalies: List[Dict]) -> List[Dict]:
        """Identifie les candidats pour fact-checking (CORRIGÉ - enrichissement proactif)."""
        candidates = []
        
        # Entités suspectes détectées
        for entity in entity_issues:
            candidates.append({
                'type': 'entity', 
                'text': entity.get('entity', ''), 
                'priority': 0.8, 
                'check_method': 'external_database'
            })
        
        # NOUVEAU : Extraction proactive d'entités importantes pour enrichissement
        if hasattr(self, 'nlp') and hasattr(self, '_current_text'):
            doc = self.nlp(self._current_text)
            for ent in doc.ents:
                if (ent.label_ in ['PERSON', 'ORG', 'GPE'] and 
                    len(ent.text.strip()) > 2 and 
                    not any(c['text'] == ent.text for c in candidates)):
                    candidates.append({
                        'type': 'entity',
                        'text': ent.text,
                        'priority': 0.6,  # Priorité standard pour enrichissement
                        'check_method': 'external_database'
                    })
        
        # Anomalies temporelles
        for temporal in temporal_anomalies:
            candidates.append({
                'type': 'temporal', 
                'text': temporal.get('text_span', ''), 
                'priority': 0.9, 
                'check_method': 'date_validation'
            })
        
        # Relations causales
        for causal in causal_anomalies:
            candidates.append({
                'type': 'causal', 
                'text': f"{causal.get('cause', '')} → {causal.get('effect', '')}", 
                'priority': 0.6, 
                'check_method': 'logical_validation'
            })
        
        # Limiter le nombre pour éviter la surcharge
        return candidates[:15]  # Top 15 candidats
    
    def _generate_validation_hints(self, text: str, entity_issues: List[Dict], temporal_anomalies: List[Dict]) -> Dict:
        """Génère les indices de validation (CORRIGÉ - enrichissement proactif)."""
        words = text.split()
        
        # Extraction proactive d'entités pour suggestions Wikidata
        wikidata_suggestions = [{'entity': entity.get('entity', ''), 'query_type': entity.get('type')} for entity in entity_issues]
        
        # Ajouter les entités importantes détectées
        if hasattr(self, 'nlp'):
            doc = self.nlp(text)
            for ent in doc.ents:
                if (ent.label_ in ['PERSON', 'ORG', 'GPE'] and 
                    len(ent.text.strip()) > 2 and 
                    not any(s['entity'] == ent.text for s in wikidata_suggestions)):
                    wikidata_suggestions.append({
                        'entity': ent.text,
                        'query_type': 'proactive_validation'
                    })
        
        return {
            'wikidata_queries': wikidata_suggestions[:10],  # Limiter à 10
            'date_checks': [{'element': temporal.get('text_span', ''), 'check_type': 'chronological_validation'} for temporal in temporal_anomalies],
            'confidence_factors': {
                'entity_density': len(entity_issues) / max(1, len(words)) * 100,
                'temporal_consistency': len(temporal_anomalies) == 0,
                'text_length': len(words)
            }
        }
    
    def get_statistics(self, results: List[HeuristicResult]) -> Dict:
        """Calcule des statistiques détaillées sur les résultats d'analyse."""
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        
        # Collecte des types de problèmes
        all_issues = []
        risk_levels = [r.risk_level for r in results]
        confidence_scores = [r.confidence_score for r in results]
        
        for r in results:
            all_issues.extend(r.detected_issues)
        
        issue_counts = Counter(issue.split(':')[0] for issue in all_issues)
        
        # Statistiques de performance
        avg_processing_time = sum(r.processing_time_ms for r in results) / total
        
        return {
            'total_summaries': total,
            'valid_summaries': valid,
            'rejection_rate_percent': (total - valid) / total * 100,
            'avg_processing_time_ms': avg_processing_time,
            'avg_confidence_score': sum(confidence_scores) / len(confidence_scores),
            'risk_distribution': dict(Counter(risk_levels)),
            'detected_issues': dict(issue_counts.most_common()),
            'confidence_distribution': {
                'min': min(confidence_scores),
                'max': max(confidence_scores),
                'median': sorted(confidence_scores)[len(confidence_scores)//2]
            }
        }


# Fonctions utilitaires pour usage simple
def quick_heuristic_check(text: str, strict: bool = False) -> bool:
    """
    Vérification heuristique rapide d'un seul résumé.
    
    Args:
        text: Texte à analyser
        strict: Mode strict (plus de validations externes)
        
    Returns:
        bool: True si valide, False si suspect
    """
    # Mode par défaut plus sensible
    sensitivity_mode = "strict" if strict else "balanced"
    detector = Level1HeuristicDetector(
        use_external_validation=strict,
        sensitivity_mode=sensitivity_mode
    )
    result = detector.detect_hallucinations(text)
    return result.is_valid






if __name__ == "__main__":
    # Test simple
    test_texts = [
        "Emmanuel Macron a déclaré hier que la France continuera ses efforts.",
        "Napoléon a utilisé son smartphone pour contacter ses généraux en 1805.",
        "La pluie a causé la chute de la bourse de Paris cette semaine.",
        "Le président Emmanuel Dupont a annoncé de nouvelles mesures économiques.",
        "L'accident de voiture a provoqué des embouteillages sur l'autoroute."
    ]
    
    detector = Level1HeuristicDetector()
    
    for i, text in enumerate(test_texts):
        result = detector.detect_hallucinations(text, f"test_{i}")
        print(f"\nTest {i}: {' VALIDE' if result.is_valid else 'SUSPECT'}")
        print(f"  Confiance: {result.confidence_score:.3f}")
        print(f"  Risque: {result.risk_level}")
        print(f"  Temps: {result.processing_time_ms:.1f}ms")
        print(f"  Texte: {text}")
        
        if result.detected_issues:
            print(f"  Issues: {'; '.join(result.detected_issues[:3])}")
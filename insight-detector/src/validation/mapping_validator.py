# src/validation/mapping_validator.py
"""
Système de validation des mappings articles/résumés.

Corrige le problème fondamental de résumés complètement déconnectés 
de leurs articles sources (hallucinations par confidence_weighted).

Fonctionnalités:
- Validation cohérence thématique article ↔ résumé
- Détection mappings corrompus/incohérents
- Score de confiance du mapping
- Recommandations de correction
"""

import re
import time
import spacy
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import Counter
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MappingValidationResult:
    """Résultat de validation d'un mapping article/résumé."""
    article_id: str
    summary_id: str
    is_valid_mapping: bool
    confidence_score: float
    thematic_coherence: float
    entity_overlap: float
    keyword_overlap: float
    structural_similarity: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time_ms: float
    validation_details: Dict[str, Any]


class ArticleSummaryMappingValidator:
    """
    Validateur de mappings article/résumé avec détection d'incohérences.
    
    Détecte:
    - Résumés complètement hors-sujet (hallucinations confidence_weighted)
    - Mappings corrompus (article A → résumé de l'article B)
    - Incohérences thématiques graves
    - Pollution croisée entre résumés
    """
    
    def __init__(self, 
                 enable_deep_analysis: bool = True,
                 thematic_threshold: float = 0.15,  # Seuil cohérence thématique minimum
                 entity_threshold: float = 0.05,   # Seuil overlap entités minimum
                 keyword_threshold: float = 0.10): # Seuil overlap mots-clés minimum
        """
        Initialise le validateur de mappings.
        
        Args:
            enable_deep_analysis: Active analyse approfondie (entités, structures)
            thematic_threshold: Seuil minimum cohérence thématique (15% par défaut)
            entity_threshold: Seuil minimum overlap entités (5% par défaut)
            keyword_threshold: Seuil minimum overlap mots-clés (10% par défaut)
        """
        
        self.enable_deep_analysis = enable_deep_analysis
        self.thematic_threshold = thematic_threshold
        self.entity_threshold = entity_threshold
        self.keyword_threshold = keyword_threshold
        
        # Modèle spaCy pour analyse sémantique
        self.nlp = None
        if enable_deep_analysis:
            try:
                # Essayer français d'abord, puis anglais
                try:
                    self.nlp = spacy.load("fr_core_news_sm")
                except OSError:
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.warning("Aucun modèle spaCy disponible, analyse simplifiée")
                        self.enable_deep_analysis = False
            except Exception as e:
                logger.warning(f"Erreur initialisation spaCy: {e}")
                self.enable_deep_analysis = False
        
        # Patterns de corruption mapping
        self.corruption_indicators = [
            # Signatures journalistiques différentes
            re.compile(r'Le\s+Figaro', re.IGNORECASE),
            re.compile(r'Le\s+Monde', re.IGNORECASE),
            re.compile(r'Libération', re.IGNORECASE),
            re.compile(r'Le\s+Nouvel\s+Obs', re.IGNORECASE),
            re.compile(r'France\s+Info', re.IGNORECASE),
            
            # Patterns géographiques
            re.compile(r'\b(?:Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier|Bordeaux|Lille)\b', re.IGNORECASE),
            re.compile(r'\b(?:France|Japon|Allemagne|Italie|Espagne|États-Unis|Chine|Russie)\b', re.IGNORECASE),
            
            # Patterns thématiques
            re.compile(r'\b(?:foot|football|sport|match|équipe|victoire|défaite)\b', re.IGNORECASE),
            re.compile(r'\b(?:économie|finance|bourse|inflation|PIB|croissance)\b', re.IGNORECASE),
            re.compile(r'\b(?:politique|élections|gouvernement|ministre|président)\b', re.IGNORECASE),
            re.compile(r'\b(?:santé|médecine|hôpital|traitement|vaccin|maladie)\b', re.IGNORECASE),
        ]
        
        # Cache pour optimiser performance
        self.analysis_cache = {}
        
    def validate_mapping(self, article: Dict[str, Any], summary: str, 
                        summary_metadata: Optional[Dict] = None) -> MappingValidationResult:
        """
        Valide la cohérence d'un mapping article → résumé.
        
        Args:
            article: Données article (id, text, title, url, etc.)
            summary: Texte du résumé
            summary_metadata: Métadonnées résumé (strategy, grade, etc.)
            
        Returns:
            MappingValidationResult avec diagnostic complet
        """
        
        start_time = time.time()
        
        article_id = str(article.get('id', 'unknown'))
        article_text = article.get('text', '')
        article_title = article.get('title', '')
        
        # Métadonnées par défaut
        if summary_metadata is None:
            summary_metadata = {}
        
        strategy = summary_metadata.get('strategy', 'unknown')
        summary_id = f"{article_id}_{strategy}"
        
        issues = []
        recommendations = []
        validation_details = {}
        
        # 1. Validation de base (existence, longueur)
        basic_validation = self._validate_basic_requirements(article_text, summary, article_title)
        issues.extend(basic_validation['issues'])
        validation_details.update(basic_validation['details'])
        
        # 2. Analyse cohérence thématique
        thematic_analysis = self._analyze_thematic_coherence(article_text, summary, article_title)
        thematic_coherence = thematic_analysis['coherence_score']
        issues.extend(thematic_analysis['issues'])
        validation_details.update(thematic_analysis['details'])
        
        # 3. Analyse overlap entités (si spaCy disponible)
        entity_overlap = 0.0
        if self.enable_deep_analysis and self.nlp:
            entity_analysis = self._analyze_entity_overlap(article_text, summary)
            entity_overlap = entity_analysis['overlap_score']
            issues.extend(entity_analysis['issues'])
            validation_details.update(entity_analysis['details'])
        
        # 4. Analyse overlap mots-clés
        keyword_analysis = self._analyze_keyword_overlap(article_text, summary, article_title)
        keyword_overlap = keyword_analysis['overlap_score']
        issues.extend(keyword_analysis['issues'])
        validation_details.update(keyword_analysis['details'])
        
        # 5. Analyse similarité structurelle
        structural_analysis = self._analyze_structural_similarity(article_text, summary)
        structural_similarity = structural_analysis['similarity_score']
        issues.extend(structural_analysis['issues'])
        validation_details.update(structural_analysis['details'])
        
        # 6. Détection patterns corruption spécifiques
        corruption_analysis = self._detect_mapping_corruption(article, summary, summary_metadata)
        issues.extend(corruption_analysis['issues'])
        validation_details.update(corruption_analysis['details'])
        
        # 7. Calcul score confiance global
        confidence_score = self._calculate_mapping_confidence(
            thematic_coherence, entity_overlap, keyword_overlap, 
            structural_similarity, issues
        )
        
        # 8. Décision validation finale
        is_valid_mapping = self._make_mapping_decision(
            confidence_score, thematic_coherence, entity_overlap, keyword_overlap, issues
        )
        
        # 9. Génération recommandations
        recommendations = self._generate_mapping_recommendations(
            is_valid_mapping, thematic_coherence, entity_overlap, 
            keyword_overlap, issues, strategy
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return MappingValidationResult(
            article_id=article_id,
            summary_id=summary_id,
            is_valid_mapping=is_valid_mapping,
            confidence_score=confidence_score,
            thematic_coherence=thematic_coherence,
            entity_overlap=entity_overlap,
            keyword_overlap=keyword_overlap,
            structural_similarity=structural_similarity,
            issues=issues,
            recommendations=recommendations,
            processing_time_ms=processing_time,
            validation_details=validation_details
        )
    
    def _validate_basic_requirements(self, article_text: str, summary: str, 
                                   article_title: str) -> Dict[str, Any]:
        """Validation des exigences de base."""
        
        issues = []
        details = {}
        
        # Longueurs
        article_words = len(article_text.split())
        summary_words = len(summary.split())
        title_words = len(article_title.split())
        
        details['article_word_count'] = article_words
        details['summary_word_count'] = summary_words
        details['title_word_count'] = title_words
        
        # Vérifications de base
        if not article_text.strip():
            issues.append({
                'type': 'article_vide',
                'severity': 'critical',
                'description': 'Article source vide',
                'confidence': 1.0
            })
        
        if not summary.strip():
            issues.append({
                'type': 'resume_vide',
                'severity': 'critical', 
                'description': 'Résumé vide',
                'confidence': 1.0
            })
        
        if article_words < 50:
            issues.append({
                'type': 'article_trop_court',
                'severity': 'moderate',
                'description': f'Article très court: {article_words} mots',
                'confidence': 0.8
            })
        
        if summary_words < 10:
            issues.append({
                'type': 'resume_trop_court',
                'severity': 'moderate',
                'description': f'Résumé très court: {summary_words} mots',
                'confidence': 0.8
            })
        
        # Ratio longueur (résumé vs article)
        if article_words > 0:
            length_ratio = summary_words / article_words
            details['length_ratio'] = length_ratio
            
            if length_ratio > 0.8:  # Résumé trop long vs article
                issues.append({
                    'type': 'ratio_longueur_suspect',
                    'severity': 'moderate',
                    'description': f'Résumé très long vs article: {length_ratio:.1%}',
                    'confidence': 0.7
                })
        
        return {
            'issues': issues,
            'details': details
        }
    
    def _analyze_thematic_coherence(self, article_text: str, summary: str, 
                                  article_title: str) -> Dict[str, Any]:
        """Analyse cohérence thématique avec détection hallucinations."""
        
        issues = []
        details = {}
        
        # Extraction mots significatifs (méthode améliorée)
        def extract_thematic_words(text: str, min_length: int = 4) -> Set[str]:
            """Extraction mots thématiques avec filtrage avancé."""
            
            # Nettoyage et tokenisation
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{' + str(min_length) + ',}\b', text.lower())
            
            # Stop words étendus (français + anglais + mots vides)
            stop_words = {
                # Français
                'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans', 'sur', 'avec', 'pour',
                'par', 'sans', 'sous', 'vers', 'chez', 'depuis', 'pendant', 'contre', 'entre',
                'que', 'qui', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi',
                'ce', 'cette', 'ces', 'cet', 'celui', 'celle', 'ceux', 'celles',
                'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
                'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
                'me', 'te', 'se', 'nous', 'vous',
                'moi', 'toi', 'lui', 'elle', 'nous', 'vous', 'eux', 'elles',
                'avoir', 'être', 'faire', 'aller', 'venir', 'voir', 'savoir', 'pouvoir', 'vouloir', 'devoir',
                'tout', 'tous', 'toute', 'toutes', 'autre', 'autres', 'même', 'mêmes',
                'plus', 'moins', 'très', 'trop', 'assez', 'aussi', 'encore', 'déjà', 'jamais', 'toujours',
                'bien', 'mal', 'mieux', 'pire', 'beaucoup', 'peu', 'plusieurs',
                'premier', 'première', 'dernier', 'dernière', 'prochain', 'prochaine',
                'grand', 'grande', 'petit', 'petite', 'gros', 'grosse', 'nouveau', 'nouvelle', 'vieux', 'vieille',
                'selon', 'après', 'avant', 'pendant', 'durant', 'lors',
                
                # Anglais
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
                'very', 'too', 'quite', 'rather', 'much', 'many', 'most', 'more', 'less', 'some', 'any',
                'all', 'both', 'each', 'few', 'other', 'another', 'such', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now',
                
                # Mots vides techniques/journalistiques
                'article', 'résumé', 'selon', 'rapport', 'étude', 'analyse', 'recherche',
                'information', 'données', 'chiffres', 'statistiques', 'pourcentage',
                'déclaration', 'annonce', 'communiqué', 'interview', 'propos'
            }
            
            return set(word for word in words if word not in stop_words)
        
        # Extraction mots thématiques
        article_words = extract_thematic_words(article_text + ' ' + article_title)
        summary_words = extract_thematic_words(summary)
        
        details['article_thematic_words'] = len(article_words)
        details['summary_thematic_words'] = len(summary_words)
        
        # Calcul overlap thématique (Jaccard)
        if not article_words or not summary_words:
            coherence_score = 0.0
            details['thematic_overlap_method'] = 'empty_sets'
        else:
            intersection = len(article_words & summary_words)
            union = len(article_words | summary_words)
            coherence_score = intersection / union if union > 0 else 0.0
            
            details['thematic_intersection'] = intersection
            details['thematic_union'] = union
            details['thematic_overlap_method'] = 'jaccard'
            details['common_words'] = list(article_words & summary_words)[:10]  # Top 10 exemples
        
        details['thematic_coherence_score'] = coherence_score
        
        # Détection hallucination complète
        if coherence_score < 0.02:  # <2% = hallucination probable
            issues.append({
                'type': 'hallucination_thematique_complete',
                'severity': 'critical',
                'description': f'Aucune cohérence thématique détectée: {coherence_score:.1%}',
                'confidence': 0.95,
                'value': coherence_score
            })
        elif coherence_score < self.thematic_threshold:
            issues.append({
                'type': 'coherence_thematique_faible',
                'severity': 'moderate',
                'description': f'Cohérence thématique faible: {coherence_score:.1%}',
                'confidence': 0.8,
                'value': coherence_score
            })
        
        # Analyse titre vs résumé (souvent plus révélateur)
        if article_title:
            title_words = extract_thematic_words(article_title)
            if title_words and summary_words:
                title_overlap = len(title_words & summary_words) / len(title_words | summary_words)
                details['title_overlap'] = title_overlap
                
                if title_overlap < 0.1 and coherence_score < 0.1:  # Double confirmation hallucination
                    issues.append({
                        'type': 'titre_resume_incoherent',
                        'severity': 'critical',
                        'description': f'Titre et résumé complètement incohérents',
                        'confidence': 0.9,
                        'value': title_overlap
                    })
        
        return {
            'coherence_score': coherence_score,
            'issues': issues,
            'details': details
        }
    
    def _analyze_entity_overlap(self, article_text: str, summary: str) -> Dict[str, Any]:
        """Analyse overlap entités nommées avec spaCy."""
        
        issues = []
        details = {}
        overlap_score = 0.0
        
        if not self.nlp:
            return {
                'overlap_score': 0.0,
                'issues': [],
                'details': {'spacy_available': False}
            }
        
        try:
            # Extraction entités
            article_doc = self.nlp(article_text[:1000000])  # Limite pour performance
            summary_doc = self.nlp(summary)
            
            # Entités normalisées (texte lower + label)
            article_entities = set(
                (ent.text.lower(), ent.label_) 
                for ent in article_doc.ents 
                if len(ent.text.strip()) > 2
            )
            summary_entities = set(
                (ent.text.lower(), ent.label_) 
                for ent in summary_doc.ents 
                if len(ent.text.strip()) > 2
            )
            
            details['article_entities_count'] = len(article_entities)
            details['summary_entities_count'] = len(summary_entities)
            
            # Calcul overlap
            if article_entities and summary_entities:
                intersection = len(article_entities & summary_entities)
                union = len(article_entities | summary_entities)
                overlap_score = intersection / union if union > 0 else 0.0
                
                details['entity_intersection'] = intersection
                details['entity_union'] = union
                details['common_entities'] = list(article_entities & summary_entities)[:5]
            
            details['entity_overlap_score'] = overlap_score
            
            # Détection problèmes entités
            if overlap_score < self.entity_threshold and len(summary_entities) > 2:
                issues.append({
                    'type': 'entites_non_coherentes',
                    'severity': 'moderate',
                    'description': f'Entités non cohérentes entre article et résumé: {overlap_score:.1%}',
                    'confidence': 0.7,
                    'value': overlap_score
                })
            
            # Détection entités "alien" (présentes dans résumé mais pas dans article)
            alien_entities = summary_entities - article_entities
            if len(alien_entities) > len(summary_entities) * 0.7:  # >70% entités "alien"
                issues.append({
                    'type': 'entites_aliens_majoritaires',
                    'severity': 'critical',
                    'description': f'Majorité entités du résumé absentes de l\'article: {len(alien_entities)}/{len(summary_entities)}',
                    'confidence': 0.85,
                    'details': {'alien_entities': list(alien_entities)[:3]}
                })
        
        except Exception as e:
            logger.warning(f"Erreur analyse entités: {e}")
            details['spacy_error'] = str(e)
        
        return {
            'overlap_score': overlap_score,
            'issues': issues,
            'details': details
        }
    
    def _analyze_keyword_overlap(self, article_text: str, summary: str, 
                               article_title: str) -> Dict[str, Any]:
        """Analyse overlap mots-clés avec pondération intelligente."""
        
        issues = []
        details = {}
        
        # Extraction mots-clés pondérés (titre = 3x, début article = 2x, reste = 1x)
        def extract_weighted_keywords(text: str, weight: float = 1.0) -> Counter:
            """Extraction mots-clés avec pondération."""
            
            # Mots significatifs
            words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
            
            # Filtrage stop words basiques
            stop_words = {
                'le', 'la', 'les', 'un', 'une', 'des', 'dans', 'sur', 'avec', 'pour', 'par',
                'que', 'qui', 'dont', 'où', 'ce', 'cette', 'cette', 'être', 'avoir', 'faire',
                'tout', 'tous', 'plus', 'très', 'bien', 'selon', 'après', 'avant', 'pendant',
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was',
                'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'new',
                'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'may', 'she', 'use'
            }
            
            filtered_words = [word for word in words if word not in stop_words and len(word) >= 4]
            
            # Comptage avec pondération
            weighted_count = Counter()
            for word in filtered_words:
                weighted_count[word] += weight
            
            return weighted_count
        
        # Extraction avec pondérations
        title_keywords = extract_weighted_keywords(article_title, weight=3.0)
        article_start = article_text[:500] if len(article_text) > 500 else article_text
        article_start_keywords = extract_weighted_keywords(article_start, weight=2.0)
        article_rest_keywords = extract_weighted_keywords(article_text[500:], weight=1.0)
        summary_keywords = extract_weighted_keywords(summary, weight=1.0)
        
        # Fusion mots-clés article avec pondérations
        article_keywords = title_keywords + article_start_keywords + article_rest_keywords
        
        details['article_keywords_count'] = len(article_keywords)
        details['summary_keywords_count'] = len(summary_keywords)
        
        # Calcul overlap pondéré
        if not article_keywords or not summary_keywords:
            overlap_score = 0.0
        else:
            # Score basé sur poids communs
            common_weight = sum(
                min(article_keywords[word], summary_keywords[word])
                for word in set(article_keywords.keys()) & set(summary_keywords.keys())
            )
            total_weight = sum(article_keywords.values()) + sum(summary_keywords.values())
            overlap_score = (2 * common_weight) / total_weight if total_weight > 0 else 0.0
        
        details['keyword_overlap_score'] = overlap_score
        details['common_keywords'] = list(
            (set(article_keywords.keys()) & set(summary_keywords.keys()))
        )[:10]
        
        # Détection problèmes mots-clés
        if overlap_score < self.keyword_threshold:
            severity = 'critical' if overlap_score < 0.05 else 'moderate'
            issues.append({
                'type': 'mots_cles_non_coherents',
                'severity': severity,
                'description': f'Mots-clés non cohérents: {overlap_score:.1%}',
                'confidence': 0.8,
                'value': overlap_score
            })
        
        return {
            'overlap_score': overlap_score,
            'issues': issues,
            'details': details
        }
    
    def _analyze_structural_similarity(self, article_text: str, summary: str) -> Dict[str, Any]:
        """Analyse similarité structurelle (longueurs, patterns, etc.)."""
        
        issues = []
        details = {}
        
        # Métriques structurelles
        article_sentences = len([s for s in re.split(r'[.!?]+', article_text) if s.strip()])
        summary_sentences = len([s for s in re.split(r'[.!?]+', summary) if s.strip()])
        
        article_words = len(article_text.split())
        summary_words = len(summary.split())
        
        details['article_sentences'] = article_sentences
        details['summary_sentences'] = summary_sentences
        details['article_words'] = article_words
        details['summary_words'] = summary_words
        
        # Ratios structurels
        sentence_compression = summary_sentences / max(1, article_sentences)
        word_compression = summary_words / max(1, article_words)
        
        details['sentence_compression'] = sentence_compression
        details['word_compression'] = word_compression
        
        # Similarité structurelle composite
        # Bon résumé: compression cohérente, pas trop extrême
        optimal_word_compression = 0.15  # 15% mots conservés
        optimal_sentence_compression = 0.25  # 25% phrases conservées
        
        word_compression_score = 1.0 - abs(word_compression - optimal_word_compression) / optimal_word_compression
        sentence_compression_score = 1.0 - abs(sentence_compression - optimal_sentence_compression) / optimal_sentence_compression
        
        similarity_score = max(0.0, (word_compression_score + sentence_compression_score) / 2)
        
        details['structural_similarity_score'] = similarity_score
        
        # Détection anomalies structurelles
        if word_compression > 0.8:  # Résumé trop long vs article
            issues.append({
                'type': 'resume_disproportionne_long',
                'severity': 'moderate',
                'description': f'Résumé anormalement long vs article: {word_compression:.1%}',
                'confidence': 0.7,
                'value': word_compression
            })
        
        if word_compression < 0.02:  # Résumé trop court vs article
            issues.append({
                'type': 'resume_disproportionne_court',
                'severity': 'minor',
                'description': f'Résumé très court vs article: {word_compression:.1%}',
                'confidence': 0.5,
                'value': word_compression
            })
        
        return {
            'similarity_score': similarity_score,
            'issues': issues,
            'details': details
        }
    
    def _detect_mapping_corruption(self, article: Dict, summary: str, 
                                 summary_metadata: Dict) -> Dict[str, Any]:
        """Détection corruption mapping avec patterns spécifiques."""
        
        issues = []
        details = {}
        
        article_text = article.get('text', '')
        article_title = article.get('title', '')
        article_url = article.get('url', '')
        strategy = summary_metadata.get('strategy', '')
        
        # Détection signatures journalistiques croisées
        article_patterns = []
        summary_patterns = []
        
        for pattern in self.corruption_indicators:
            if pattern.search(article_text + ' ' + article_title):
                article_patterns.append(pattern.pattern)
            if pattern.search(summary):
                summary_patterns.append(pattern.pattern)
        
        details['article_signature_patterns'] = article_patterns
        details['summary_signature_patterns'] = summary_patterns
        
        # Détection mapping croisé (signatures différentes)
        different_signatures = set(summary_patterns) - set(article_patterns)
        if different_signatures and len(article_patterns) > 0:
            issues.append({
                'type': 'signatures_journalistiques_croisees',
                'severity': 'critical',
                'description': f'Signatures journalistiques différentes détectées',
                'confidence': 0.9,
                'details': {
                    'article_patterns': article_patterns,
                    'summary_patterns': list(different_signatures)
                }
            })
        
        # Détection corruption confidence_weighted spécifique
        if strategy == "confidence_weighted":
            # Pattern signature typique corruption
            cw_corruption_pattern = re.compile(r'Par\s+Le\s+Nouvel\s+Obs\s+avec\s+[^\\w\\s]', re.IGNORECASE)
            if cw_corruption_pattern.search(summary):
                issues.append({
                    'type': 'corruption_confidence_weighted_mapping',
                    'severity': 'critical',
                    'description': 'Corruption confidence_weighted détectée dans mapping',
                    'confidence': 0.98
                })
        
        # Vérification cohérence URL/contenu (si disponible)
        if article_url:
            url_domain = re.search(r'//([^/]+)', article_url)
            if url_domain:
                domain = url_domain.group(1).lower()
                details['article_domain'] = domain
                
                # Vérification cohérence domaine/contenu
                domain_patterns = {
                    'lefigaro': r'Le\s+Figaro',
                    'lemonde': r'Le\s+Monde', 
                    'liberation': r'Libération',
                    'nouvelobs': r'Le\s+Nouvel\s+Obs',
                    'franceinfo': r'France\s+Info'
                }
                
                for domain_key, pattern in domain_patterns.items():
                    if domain_key in domain and not re.search(pattern, summary, re.IGNORECASE):
                        # Domaine attendu mais signature différente dans résumé
                        summary_source = None
                        for other_pattern in domain_patterns.values():
                            if re.search(other_pattern, summary, re.IGNORECASE):
                                summary_source = other_pattern
                                break
                        
                        if summary_source:
                            issues.append({
                                'type': 'incoherence_domaine_signature',
                                'severity': 'critical',
                                'description': f'Domaine article ({domain}) incohérent avec signature résumé',
                                'confidence': 0.85,
                                'details': {
                                    'expected_pattern': pattern,
                                    'found_pattern': summary_source
                                }
                            })
        
        return {
            'issues': issues,
            'details': details
        }
    
    def _calculate_mapping_confidence(self, thematic_coherence: float, entity_overlap: float,
                                    keyword_overlap: float, structural_similarity: float,
                                    issues: List[Dict]) -> float:
        """Calcul score confiance mapping composite."""
        
        # Pondérations selon importance
        weights = {
            'thematic': 0.4,    # Plus important
            'keyword': 0.3,     # Important
            'entity': 0.2,      # Modéré (peut être manquant)
            'structural': 0.1   # Moins important
        }
        
        # Score de base pondéré
        base_score = (
            thematic_coherence * weights['thematic'] +
            keyword_overlap * weights['keyword'] +
            entity_overlap * weights['entity'] +
            structural_similarity * weights['structural']
        )
        
        # Pénalités selon sévérité issues
        penalty = 0.0
        for issue in issues:
            severity = issue.get('severity', 'minor')
            if severity == 'critical':
                penalty += 0.3
            elif severity == 'moderate':
                penalty += 0.15
            else:  # minor
                penalty += 0.05
        
        # Score final
        confidence_score = max(0.0, min(1.0, base_score - penalty))
        
        return confidence_score
    
    def _make_mapping_decision(self, confidence_score: float, thematic_coherence: float,
                             entity_overlap: float, keyword_overlap: float,
                             issues: List[Dict]) -> bool:
        """Décision validation mapping finale."""
        
        # Rejet automatique si problèmes critiques
        critical_issues = [i for i in issues if i.get('severity') == 'critical']
        if len(critical_issues) >= 1:
            return False
        
        # Rejet si cohérence thématique trop faible (hallucination)
        if thematic_coherence < self.thematic_threshold:
            return False
        
        # Rejet si overlap mots-clés trop faible 
        if keyword_overlap < self.keyword_threshold:
            return False
        
        # Score confiance minimum requis
        if confidence_score < 0.5:
            return False
        
        return True
    
    def _generate_mapping_recommendations(self, is_valid: bool, thematic_coherence: float,
                                        entity_overlap: float, keyword_overlap: float,
                                        issues: List[Dict], strategy: str) -> List[str]:
        """Génération recommandations selon problèmes détectés."""
        
        recommendations = []
        
        if not is_valid:
            # Recommandations selon type de problème
            critical_issues = [i for i in issues if i.get('severity') == 'critical']
            
            for issue in critical_issues:
                issue_type = issue['type']
                
                if 'hallucination' in issue_type or 'coherence' in issue_type:
                    recommendations.append("Régénérer résumé depuis texte source original")
                    recommendations.append("Vérifier mapping article/résumé dans dataset")
                
                elif 'corruption_confidence_weighted' in issue_type:
                    recommendations.append("Remplacer par stratégie non-confidence_weighted")
                    recommendations.append("Régénérer avec stratégie alternative")
                
                elif 'signatures_croisees' in issue_type or 'incoherence_domaine' in issue_type:
                    recommendations.append("Vérifier intégrité dataset - mapping possiblement corrompu")
                    recommendations.append("Régénérer à partir source vérifiée")
            
            # Recommandations générales selon métriques
            if thematic_coherence < 0.1:
                recommendations.append("Cohérence thématique critique - régénération obligatoire")
            
            if keyword_overlap < 0.05:
                recommendations.append("Overlap mots-clés inexistant - vérifier mapping")
            
            if strategy == "confidence_weighted":
                recommendations.append("Éviter stratégie confidence_weighted (problèmes récurrents)")
        
        else:
            # Améliorations pour mappings valides mais perfectibles
            if thematic_coherence < 0.3:
                recommendations.append("Améliorer cohérence thématique")
            
            if entity_overlap < 0.1 and entity_overlap > 0:
                recommendations.append("Enrichir résumé avec entités clés de l'article")
        
        # Dédoublonnage
        return list(dict.fromkeys(recommendations))
    
    def validate_batch(self, articles: List[Dict], summaries_data: Dict, 
                      enable_progress: bool = True) -> Tuple[List[MappingValidationResult], Dict[str, Any]]:
        """
        Validation batch mappings article/résumé.
        
        Args:
            articles: Liste articles sources
            summaries_data: Données résumés (format all_summaries_production.json)
            enable_progress: Affichage progression
            
        Returns:
            Tuple (résultats_validation, statistiques)
        """
        
        # Index articles par ID
        articles_by_id = {str(article['id']): article for article in articles}
        
        results = []
        total_mappings = 0
        
        # Comptage total pour progression
        for article_id, article_data in summaries_data.items():
            if 'strategies' in article_data:
                total_mappings += len(article_data['strategies'])
        
        processed = 0
        
        for article_id, article_data in summaries_data.items():
            if 'strategies' not in article_data:
                continue
            
            # Récupération article source
            if article_id not in articles_by_id:
                logger.warning(f"Article {article_id} introuvable dans sources")
                continue
            
            article = articles_by_id[article_id]
            
            for strategy, strategy_data in article_data['strategies'].items():
                processed += 1
                
                if enable_progress and processed % 50 == 0:
                    logger.info(f"Validation mappings: {processed}/{total_mappings} ({processed/total_mappings:.1%})")
                
                summary = strategy_data.get('summary', '')
                summary_metadata = {
                    'strategy': strategy,
                    'article_id': article_id,
                    'coherence': strategy_data.get('coherence'),
                    'factuality': strategy_data.get('factuality'),
                    'original_grade': strategy_data.get('original_grade')
                }
                
                # Validation mapping
                result = self.validate_mapping(article, summary, summary_metadata)
                results.append(result)
        
        # Calcul statistiques
        statistics = self._calculate_validation_statistics(results)
        
        if enable_progress:
            logger.info(f"Validation mappings terminée: {len(results)} mappings validés")
        
        return results, statistics
    
    def _calculate_validation_statistics(self, results: List[MappingValidationResult]) -> Dict[str, Any]:
        """Calcul statistiques validation mappings."""
        
        if not results:
            return {}
        
        total = len(results)
        valid_mappings = sum(1 for r in results if r.is_valid_mapping)
        
        # Distribution scores
        confidence_scores = [r.confidence_score for r in results]
        thematic_scores = [r.thematic_coherence for r in results]
        entity_scores = [r.entity_overlap for r in results]
        keyword_scores = [r.keyword_overlap for r in results]
        
        # Problèmes détectés
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        issue_types = Counter(issue['type'] for issue in all_issues)
        severity_counts = Counter(issue['severity'] for issue in all_issues)
        
        # Temps traitement
        avg_processing_time = sum(r.processing_time_ms for r in results) / total
        
        # Problèmes critiques par type
        critical_mappings = [r for r in results if not r.is_valid_mapping]
        critical_reasons = []
        for result in critical_mappings:
            critical_issues = [i['type'] for i in result.issues if i.get('severity') == 'critical']
            critical_reasons.extend(critical_issues)
        
        return {
            'summary': {
                'total_mappings': total,
                'valid_mappings': valid_mappings,
                'invalid_mappings': total - valid_mappings,
                'validation_rate': valid_mappings / total * 100,
                'avg_processing_time_ms': avg_processing_time
            },
            'score_statistics': {
                'confidence': {
                    'mean': np.mean(confidence_scores),
                    'median': np.median(confidence_scores),
                    'std': np.std(confidence_scores),
                    'min': min(confidence_scores),
                    'max': max(confidence_scores)
                },
                'thematic_coherence': {
                    'mean': np.mean(thematic_scores),
                    'median': np.median(thematic_scores),
                    'min': min(thematic_scores),
                    'max': max(thematic_scores)
                },
                'entity_overlap': {
                    'mean': np.mean(entity_scores),
                    'median': np.median(entity_scores)
                },
                'keyword_overlap': {
                    'mean': np.mean(keyword_scores), 
                    'median': np.median(keyword_scores)
                }
            },
            'issues_detected': {
                'total_issues': len(all_issues),
                'issue_types': dict(issue_types.most_common(15)),
                'severity_distribution': dict(severity_counts),
                'critical_reasons': dict(Counter(critical_reasons).most_common(10))
            },
            'thresholds_used': {
                'thematic_threshold': self.thematic_threshold,
                'entity_threshold': self.entity_threshold,
                'keyword_threshold': self.keyword_threshold
            },
            'quality_assessment': {
                'excellent_mappings': sum(1 for r in results if r.confidence_score >= 0.8),
                'good_mappings': sum(1 for r in results if 0.6 <= r.confidence_score < 0.8),
                'moderate_mappings': sum(1 for r in results if 0.4 <= r.confidence_score < 0.6),
                'poor_mappings': sum(1 for r in results if r.confidence_score < 0.4)
            }
        }


# Fonction utilitaire pour migration
def create_mapping_validator() -> ArticleSummaryMappingValidator:
    """Crée un validateur de mappings avec configuration optimale."""
    
    return ArticleSummaryMappingValidator(
        enable_deep_analysis=True,
        thematic_threshold=0.15,  # 15% cohérence minimum
        entity_threshold=0.05,    # 5% entités communes minimum
        keyword_threshold=0.10    # 10% mots-clés communs minimum
    )
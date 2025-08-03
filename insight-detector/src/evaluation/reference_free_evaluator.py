# src/evaluation/reference_free_evaluator.py
"""
Évaluateur sans référence pour les résumés.
Utilise des métriques qui ne nécessitent pas de résumé de référence humain.
"""

import logging
from typing import Dict, Optional, List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class ReferenceFreeEvaluator:
    """
    Évaluateur de résumés sans référence humaine.
    Utilise des métriques intrinsèques et de similarité sémantique.
    """
    
    def __init__(self, device: str = "cpu", lang: str = "fr"):
        self.device = device
        self.lang = lang
        self._load_models()
    
    def _load_models(self):
        """Charge les modèles nécessaires pour l'évaluation"""
        try:
            # Modèle multilingue pour similarité sémantique
            self.sentence_model = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device=self.device
            )
            logger.info("Modèle de similarité sémantique chargé")
        except Exception as e:
            logger.error(f"Erreur chargement modèle de similarité: {e}")
            self.sentence_model = None
    
    def evaluate_summary(
        self, 
        summary: str, 
        source: str,
        compute_similarity: bool = True,
        compute_coverage: bool = True,
        compute_intrinsic: bool = True
    ) -> Dict:
        """
        Évalue un résumé sans référence humaine.
        
        Args:
            summary: Résumé à évaluer
            source: Texte source
            compute_similarity: Calculer similarité sémantique
            compute_coverage: Calculer couverture du contenu
            compute_intrinsic: Calculer métriques intrinsèques
            
        Returns:
            Dict avec scores d'évaluation
        """
        results = {}
        
        if compute_intrinsic:
            results.update(self._compute_intrinsic_metrics(summary, source))
        
        if compute_similarity and self.sentence_model:
            results.update(self._compute_semantic_similarity(summary, source))
        
        if compute_coverage:
            results.update(self._compute_content_coverage(summary, source))
        
        # Score composite sans référence
        results['composite_score_ref_free'] = self._compute_composite_score(results)
        
        return results
    
    def _compute_intrinsic_metrics(self, summary: str, source: str) -> Dict:
        """Calcule des métriques intrinsèques au résumé"""
        metrics = {}
        
        # Métriques de base
        summary_words = summary.split()
        source_words = source.split()
        
        metrics['summary_length'] = len(summary_words)
        metrics['source_length'] = len(source_words)
        metrics['compression_ratio'] = len(summary_words) / len(source_words) if source_words else 0
        
        # Diversité lexicale
        unique_words = set(summary_words)
        metrics['lexical_diversity'] = len(unique_words) / len(summary_words) if summary_words else 0
        
        # Répétitivité (inverse de la diversité)
        metrics['repetitiveness'] = 1 - metrics['lexical_diversity']
        
        # Longueur des phrases
        sentences = summary.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        metrics['avg_sentence_length'] = avg_sentence_length
        
        # Score de lisibilité basique
        metrics['readability_basic'] = self._compute_basic_readability(summary)
        
        return metrics
    
    def _compute_semantic_similarity(self, summary: str, source: str) -> Dict:
        """Calcule la similarité sémantique entre résumé et source"""
        if not self.sentence_model:
            return {}
        
        try:
            # Encoder les textes
            summary_embedding = self.sentence_model.encode([summary])
            source_embedding = self.sentence_model.encode([source])
            
            # Similarité cosinus
            similarity = torch.nn.functional.cosine_similarity(
                torch.tensor(summary_embedding),
                torch.tensor(source_embedding)
            ).item()
            
            # Similarité par phrases (pour évaluer la cohérence)
            summary_sentences = [s.strip() for s in summary.split('.') if s.strip()]
            if len(summary_sentences) > 1:
                sentence_embeddings = self.sentence_model.encode(summary_sentences)
                
                # Cohérence interne (similarité moyenne entre phrases)
                similarities = []
                for i in range(len(sentence_embeddings)):
                    for j in range(i+1, len(sentence_embeddings)):
                        sim = torch.nn.functional.cosine_similarity(
                            torch.tensor(sentence_embeddings[i:i+1]),
                            torch.tensor(sentence_embeddings[j:j+1])
                        ).item()
                        similarities.append(sim)
                
                internal_coherence = np.mean(similarities) if similarities else 0
            else:
                internal_coherence = 1.0  # Une seule phrase = cohérente par défaut
            
            return {
                'semantic_similarity': similarity,
                'internal_coherence': internal_coherence,
                'coherence_penalty': max(0, 0.3 - internal_coherence)  # Pénalité si < 0.3
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul similarité sémantique: {e}")
            return {}
    
    def _compute_content_coverage(self, summary: str, source: str) -> Dict:
        """Évalue la couverture du contenu source"""
        
        # Extraction des entités et concepts clés
        summary_words = set(summary.lower().split())
        source_words = set(source.lower().split())
        
        # Couverture lexicale
        intersection = summary_words.intersection(source_words)
        coverage = len(intersection) / len(source_words) if source_words else 0
        
        # Informativité (mots uniques du résumé)
        informativeness = len(summary_words - source_words) / len(summary_words) if summary_words else 0
        
        # Extraction d'entités importantes (noms propres, chiffres)
        import re
        
        # Entités nommées basiques (noms propres)
        source_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', source))
        summary_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary))
        
        entity_coverage = (
            len(summary_entities.intersection(source_entities)) / len(source_entities)
            if source_entities else 0
        )
        
        # Chiffres et dates
        source_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)*\b', source))
        summary_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)*\b', summary))
        
        number_coverage = (
            len(summary_numbers.intersection(source_numbers)) / len(source_numbers)
            if source_numbers else 0
        )
        
        return {
            'content_coverage': coverage,
            'informativeness': informativeness,
            'entity_coverage': entity_coverage,
            'number_coverage': number_coverage,
            'coverage_score': np.mean([coverage, entity_coverage, number_coverage])
        }
    
    def _compute_basic_readability(self, text: str) -> float:
        """Calcule un score de lisibilité basique"""
        if not text:
            return 0
        
        words = text.split()
        sentences = text.split('.')
        
        # Métriques basiques
        avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
        avg_chars_per_word = np.mean([len(word) for word in words]) if words else 0
        
        # Score simplifié (inverse de la complexité)
        complexity = (avg_words_per_sentence / 20) + (avg_chars_per_word / 10)
        readability = max(0, 1 - complexity)
        
        return readability
    
    def _compute_composite_score(self, metrics: Dict) -> float:
        """Calcule un score composite sans référence"""
        
        # Pondération des métriques disponibles
        weights = {
            'semantic_similarity': 0.3,
            'internal_coherence': 0.2,
            'coverage_score': 0.25,
            'lexical_diversity': 0.1,
            'readability_basic': 0.15
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics and metrics[metric] is not None:
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        # Pénalités
        penalties = 0
        
        # Pénalité longueur inappropriée
        compression = metrics.get('compression_ratio', 0)
        if compression < 0.1 or compression > 0.8:  # Trop court ou trop long
            penalties += 0.1
        
        # Pénalité répétitivité excessive
        repetitiveness = metrics.get('repetitiveness', 0)
        if repetitiveness > 0.5:
            penalties += 0.1
        
        # Pénalité cohérence interne faible
        coherence_penalty = metrics.get('coherence_penalty', 0)
        penalties += coherence_penalty
        
        composite = (weighted_sum / total_weight if total_weight > 0 else 0) - penalties
        return max(0, min(1, composite))


# Fonction utilitaire pour intégration
def evaluate_summary_without_reference(summary: str, source: str, device: str = "cpu") -> Dict:
    """
    Fonction utilitaire pour évaluer un résumé sans référence.
    
    Args:
        summary: Résumé à évaluer
        source: Texte source
        device: Dispositif de calcul
        
    Returns:
        Dict avec métriques d'évaluation
    """
    evaluator = ReferenceFreeEvaluator(device=device)
    return evaluator.evaluate_summary(summary, source)
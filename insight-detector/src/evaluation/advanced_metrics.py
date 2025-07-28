# src/evaluation/advanced_metrics.py

import numpy as np
import re
import logging
from typing import Dict, List, Union, Set

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.model_cache import get_sentence_transformer


class AdvancedMetrics:
    """Métriques avancées : factualité, cohérence, hallucinations"""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.factuality_model = None
        self._load_models()

    def _load_models(self):
        """Chargement modèles spécialisés, mutualisation via cache"""
        try:
            self.factuality_model = get_sentence_transformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
            logger.info("Modèle factualité chargé")
        except Exception as e:
            logger.warning(f"Modèle factualité non disponible: {e}")

    def calculate_factuality_score(self, summary: str, source: str) -> Dict[str, float]:
        """
        Score de factualité [0,1] par alignement sémantique phrase à phrase.
        """
        if not self.factuality_model:
            return {'factuality_score': 0.5, 'error': 'Modèle non disponible'}

        try:
            summary_sentences = nltk.sent_tokenize(summary)
            source_sentences = nltk.sent_tokenize(source)
            if not summary_sentences or not source_sentences:
                return {'factuality_score': 0.5}

            summary_embeddings = self.factuality_model.encode(summary_sentences)
            source_embeddings = self.factuality_model.encode(source_sentences)

            factuality_scores = [
                cosine_similarity(summ_emb.reshape(1, -1), source_embeddings).max()
                for summ_emb in summary_embeddings
            ]

            mean_factuality = float(np.mean(factuality_scores))

            return {
                'factuality_score': mean_factuality,
                'sentence_factuality_scores': factuality_scores,
                'num_sentences_checked': len(summary_sentences),
                'method': 'semantic_alignment'
            }
        except Exception as e:
            logger.error(f"Erreur factualité: {e}")
            return {'factuality_score': 0.5, 'error': str(e)}

    def detect_hallucinations(self, summary: str, source: str) -> Dict[str, Union[float, List[str], int, str]]:
        """
        Détection hallucinations par entités : proportion d’entités présentes dans le résumé mais pas dans la source.
        """
        try:
            summary_entities = self._extract_simple_entities(summary)
            source_entities = self._extract_simple_entities(source)
            hallucinated_entities = summary_entities - source_entities

            hallucination_rate = (
                len(hallucinated_entities) / len(summary_entities)
                if len(summary_entities) > 0 else 0.0
            )
            return {
                'hallucination_rate': hallucination_rate,
                'hallucinated_entities': list(hallucinated_entities),
                'summary_entities_count': len(summary_entities),
                'source_entities_count': len(source_entities),
                'method': 'entity_based'
            }
        except Exception as e:
            logger.error(f"Erreur détection hallucinations: {e}")
            return {'hallucination_rate': 0.0, 'error': str(e)}

    def _extract_simple_entities(self, text: str) -> Set[str]:
        """
        Extraction rapide d’entités par patterns (majuscules, chiffres, dates).
        """
        entities = set(re.findall(r'\b[A-ZÀ-ÿ][a-zà-ÿ]+\b', text))
        numbers = set(re.findall(r'\b\d{1,4}\b', text))
        entities.update(numbers)
        common_words = {'Le', 'La', 'Les', 'Un', 'Une', 'Des', 'Ce', 'Cette', 'Ces', 'Il', 'Elle', 'Ils', 'Elles'}
        return entities - common_words

    def calculate_coherence_score(self, summary: str) -> Dict[str, float]:
        """
        Score de cohérence [0,1] : moyenne des similarités sémantiques et lexicales entre phrases consécutives.
        """
        try:
            sentences = nltk.sent_tokenize(summary)
            if len(sentences) < 2:
                return {'coherence_score': 1.0, 'method': 'single_sentence'}

            if self.factuality_model:
                embeddings = self.factuality_model.encode(sentences)
                consecutive_similarities = [
                    cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0][0]
                    for i in range(len(embeddings) - 1)
                ]
                coherence_semantic = float(np.mean(consecutive_similarities))
            else:
                coherence_semantic = 0.5

            shared_words_scores = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i+1].lower().split())
                if words1 and words2:
                    jaccard = len(words1 & words2) / len(words1 | words2)
                else:
                    jaccard = 0.0
                shared_words_scores.append(jaccard)

            coherence_lexical = float(np.mean(shared_words_scores)) if shared_words_scores else 0.5
            composite_coherence = (coherence_semantic + coherence_lexical) / 2

            return {
                'coherence_score': composite_coherence,
                'semantic_coherence': coherence_semantic,
                'lexical_coherence': coherence_lexical,
                'num_sentences': len(sentences)
            }
        except Exception as e:
            logger.error(f"Erreur cohérence: {e}")
            return {'coherence_score': 0.5, 'error': str(e)}

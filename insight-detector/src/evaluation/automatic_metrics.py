# src/evaluation/automatic_metrics.py

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from rouge_score import rouge_scorer
from bert_score import score as bert_score

from utils.model_cache import get_sentence_transformer

# NLTK + dépendances externes
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installez rouge-score: pip install rouge-score")

try:
    from bert_score import score as bert_score
except ImportError:
    print("Installez bert-score: pip install bert-score")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomaticMetrics:
    """
    Métriques automatiques standards (ROUGE, BERTScore, METEOR, abstractiveness, compression).
    """
    def __init__(self, device: str = 'cpu', lang: str = 'fr'):
        self.device = device
        self.lang = lang
        self.rouge_scorer = None
        self.semantic_model = None
        self._load_models()

    def _load_models(self):
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
            logger.info("ROUGE scorer chargé")
        except Exception as e:
            logger.warning(f"ROUGE non disponible: {e}")

        try:
            self.semantic_model = get_sentence_transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Modèle sémantique chargé")
        except Exception as e:
            logger.warning(f"Modèle sémantique non disponible: {e}")

    def calculate_rouge_scores(self, summary: str, reference: str) -> Dict[str, float]:
        """Scores ROUGE complets"""
        if not self.rouge_scorer:
            return {'error': 'ROUGE non disponible'}
        try:
            scores = self.rouge_scorer.score(reference, summary)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rouge2_p': scores['rouge2'].precision,
                'rouge2_r': scores['rouge2'].recall,
                'rougeL_f': scores['rougeL'].fmeasure,
                'rougeL_p': scores['rougeL'].precision,
                'rougeL_r': scores['rougeL'].recall,
                'rougeLsum_f': scores['rougeLsum'].fmeasure
            }
        except Exception as e:
            logger.error(f"Erreur ROUGE: {e}")
            return {'error': str(e)}

    def calculate_bert_score(self, summary: str, reference: str) -> Dict[str, float]:
        """BERTScore avec fallback sémantique."""
        try:
            P, R, F1 = bert_score([summary], [reference], lang=self.lang, device=self.device)
            return {
                'bertscore_precision': float(P[0]),
                'bertscore_recall': float(R[0]),
                'bertscore_f1': float(F1[0])
            }
        except Exception as e:
            logger.warning(f"BERTScore officiel échoué: {e}")
            # Fallback sémantique
            if self.semantic_model:
                try:
                    emb_summary = self.semantic_model.encode([summary])
                    emb_reference = self.semantic_model.encode([reference])
                    similarity = cosine_similarity(emb_summary, emb_reference)[0][0]
                    return {
                        'bertscore_precision': similarity,
                        'bertscore_recall': similarity,
                        'bertscore_f1': similarity,
                        'method': 'semantic_fallback'
                    }
                except Exception as e2:
                    logger.error(f"Fallback sémantique échoué: {e2}")
            return {'error': 'BERTScore non disponible'}

    def calculate_meteor_score(self, summary: str, reference: str) -> Dict[str, float]:
        """METEOR (fallback unigram F1 si besoin)"""
        try:
            from nltk.translate.meteor_score import meteor_score
            summary_tokens = nltk.word_tokenize(summary.lower())
            reference_tokens = nltk.word_tokenize(reference.lower())
            score = meteor_score([reference_tokens], summary_tokens)
            return {
                'meteor_score': score,
                'method': 'nltk_meteor'
            }
        except Exception as e:
            logger.warning(f"METEOR NLTK échoué: {e}")
            try:
                summary_words = set(summary.lower().split())
                reference_words = set(reference.lower().split())
                if len(reference_words) == 0:
                    return {'meteor_score': 0.0, 'method': 'fallback'}
                common_words = summary_words & reference_words
                precision = len(common_words) / len(summary_words) if summary_words else 0
                recall = len(common_words) / len(reference_words) if reference_words else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                return {'meteor_score': f1, 'method': 'unigram_f1_fallback'}
            except Exception as e2:
                logger.error(f"Fallback METEOR échoué: {e2}")
                return {'meteor_score': 0.0, 'error': str(e2)}

    def calculate_abstractiveness(self, summary: str, source: str) -> Dict[str, float]:
        """Nouveauté/abstractivité par rapport à la source"""
        try:
            summary_words = set(summary.lower().split())
            source_words = set(source.lower().split())
            if not summary_words:
                return {'abstractiveness': 0.0}
            novel_words = summary_words - source_words
            abstractiveness = len(novel_words) / len(summary_words)
            summary_bigrams = set(self._get_ngrams(summary.lower().split(), 2))
            source_bigrams = set(self._get_ngrams(source.lower().split(), 2))
            novel_bigrams = summary_bigrams - source_bigrams if summary_bigrams else set()
            bigram_abstractiveness = len(novel_bigrams) / len(summary_bigrams) if summary_bigrams else 0
            composite_score = (abstractiveness + bigram_abstractiveness) / 2
            return {
                'abstractiveness': abstractiveness,
                'bigram_abstractiveness': bigram_abstractiveness,
                'composite_abstractiveness': composite_score,
                'novel_words_count': len(novel_words),
                'novel_bigrams_count': len(novel_bigrams)
            }
        except Exception as e:
            logger.error(f"Erreur abstractiveness: {e}")
            return {'abstractiveness': 0.0, 'error': str(e)}

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def calculate_compression_ratio(self, summary: str, source: str) -> Dict[str, float]:
        """Rapport de compression longueur résumé/source"""
        summary_words = len(summary.split())
        source_words = len(source.split())
        summary_chars = len(summary)
        source_chars = len(source)
        return {
            'compression_ratio_words': summary_words / source_words if source_words > 0 else 0,
            'compression_ratio_chars': summary_chars / source_chars if source_chars > 0 else 0,
            'summary_length_words': summary_words,
            'source_length_words': source_words,
            'summary_length_chars': summary_chars,
            'source_length_chars': source_chars
        }

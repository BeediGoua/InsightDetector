# src/evaluation/business_metrics.py

import numpy as np
import re
import logging
from typing import Dict, List

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessMetrics:
    """
    Métriques business : lisibilité, engagement, densité informationnelle.
    """

    def __init__(self):
        self.readability_patterns = self._init_readability_patterns()

    def _init_readability_patterns(self) -> Dict:
        """Patterns pour analyse lisibilité française"""
        return {
            'complex_words': re.compile(r'\b\w{10,}\b'),  # Mots >10 caractères
            'passive_voice': re.compile(r'\b(être|était|été|étant)\s+\w+é\b'),
            'long_sentences': 25,  # Seuil mots par phrase
            'syllable_vowels': re.compile(r'[aeiouyàéèêëïîôöùûüÿ]', re.IGNORECASE)
        }

    def calculate_readability_score(self, text: str) -> Dict[str, float]:
        """
        Score de lisibilité adapté au français, combinant plusieurs facteurs.
        """
        try:
            sentences = nltk.sent_tokenize(text)
            words = text.split()

            if len(sentences) == 0 or len(words) == 0:
                return {'readability_score': 0.0}

            avg_sentence_length = len(words) / len(sentences)

            complex_words = len(self.readability_patterns['complex_words'].findall(text))
            complex_word_ratio = complex_words / len(words) if words else 0

            passive_sentences = len(self.readability_patterns['passive_voice'].findall(text))
            passive_ratio = passive_sentences / len(sentences) if sentences else 0

            long_sentences = sum(1 for s in sentences if len(s.split()) > self.readability_patterns['long_sentences'])
            long_sentence_ratio = long_sentences / len(sentences) if sentences else 0

            total_syllables = self._count_syllables_french(text)
            avg_syllables_per_word = total_syllables / len(words) if words else 0

            flesch_score = 207 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            flesch_normalized = max(0, min(100, flesch_score)) / 100  # Normaliser 0-1

            readability_components = {
                'flesch_component': flesch_normalized,
                'complexity_penalty': 1 - complex_word_ratio,
                'passive_penalty': 1 - passive_ratio,
                'length_penalty': 1 - long_sentence_ratio
            }
            weights = [0.4, 0.2, 0.2, 0.2]
            readability_score = sum(score * weight for score, weight in zip(readability_components.values(), weights))

            return {
                'readability_score': readability_score,
                'flesch_score_adapted': flesch_score,
                'avg_sentence_length': avg_sentence_length,
                'complex_word_ratio': complex_word_ratio,
                'passive_voice_ratio': passive_ratio,
                'long_sentence_ratio': long_sentence_ratio,
                'avg_syllables_per_word': avg_syllables_per_word,
                **readability_components
            }
        except Exception as e:
            logger.error(f"Erreur lisibilité: {e}")
            return {'readability_score': 0.5, 'error': str(e)}

    def _count_syllables_french(self, text: str) -> int:
        """Approximation : nombre de voyelles."""
        vowel_matches = self.readability_patterns['syllable_vowels'].findall(text.lower())
        return len(vowel_matches)

    def calculate_engagement_score(self, text: str) -> Dict[str, float]:
        """
        Score d'engagement basé sur longueur, diversité, mots d'action/émotion, présence de chiffres.
        """
        try:
            engagement_factors = {}
            word_count = len(text.split())
            if 150 <= word_count <= 300:
                engagement_factors['length_score'] = 1.0
            elif word_count < 150:
                engagement_factors['length_score'] = word_count / 150
            else:
                engagement_factors['length_score'] = max(0.3, 300 / word_count)

            words = text.lower().split()
            unique_words = set(words)
            ttr = len(unique_words) / len(words) if words else 0
            engagement_factors['lexical_diversity'] = min(ttr * 2, 1.0)

            action_words = ['annonce', 'révèle', 'découvre', 'lance', 'crée', 'développe', 'améliore']
            emotion_words = ['important', 'majeur', 'crucial', 'significatif', 'remarquable', 'exceptionnelle']
            action_count = sum(1 for word in action_words if word in text.lower())
            emotion_count = sum(1 for word in emotion_words if word in text.lower())
            engagement_factors['action_words_score'] = min(action_count / 3, 1.0)
            engagement_factors['emotion_words_score'] = min(emotion_count / 2, 1.0)

            numbers = re.findall(r'\b\d+\b', text)
            engagement_factors['numbers_score'] = min(len(numbers) / 3, 1.0)

            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            engagement_score = sum(score * weight for score, weight in zip(engagement_factors.values(), weights))

            return {
                'engagement_score': engagement_score,
                'word_count': word_count,
                'lexical_diversity': ttr,
                'action_words_count': action_count,
                'emotion_words_count': emotion_count,
                'numbers_count': len(numbers),
                **engagement_factors
            }
        except Exception as e:
            logger.error(f"Erreur engagement: {e}")
            return {'engagement_score': 0.5, 'error': str(e)}

    def calculate_information_density(self, summary: str, source: str) -> Dict[str, float]:
        """
        Densité d'informations : entités/longueur + conservation d'information clé.
        """
        try:
            summary_entities = len(self._extract_key_info(summary))
            source_entities = len(self._extract_key_info(source))
            summary_words = len(summary.split())
            source_words = len(source.split())
            summary_density = summary_entities / summary_words if summary_words > 0 else 0
            source_density = source_entities / source_words if source_words > 0 else 0
            info_retention = summary_entities / source_entities if source_entities > 0 else 0.5
            density_score = (summary_density + info_retention) / 2
            return {
                'information_density': density_score,
                'summary_entity_density': summary_density,
                'source_entity_density': source_density,
                'information_retention_ratio': info_retention,
                'summary_entities_count': summary_entities,
                'source_entities_count': source_entities
            }
        except Exception as e:
            logger.error(f"Erreur densité info: {e}")
            return {'information_density': 0.5, 'error': str(e)}

    def _extract_key_info(self, text: str) -> List[str]:
        """
        Extraction d'informations-clés : noms propres, nombres, dates.
        """
        key_info = []
        proper_nouns = re.findall(r'\b[A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]+)*\b', text)
        key_info.extend(proper_nouns)
        numbers = re.findall(r'\b\d+(?:[,\.]\d+)*\s*(?:%|€|dollars?|millions?|milliards?)?\b', text)
        key_info.extend(numbers)
        dates = re.findall(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b|\b\d{4}\b', text)
        key_info.extend(dates)
        return key_info

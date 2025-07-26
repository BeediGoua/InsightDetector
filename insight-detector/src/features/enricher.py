# src/features/enricher.py

import re
from typing import Dict, Optional, Union
from langdetect import detect
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

class ArticleEnricher:
    """
    Classe pour enrichir un article avec nettoyage, détection de langue, entités nommées,
    embeddings sémantiques et score de qualité.
    """
    def __init__(
        self,
        language_model: Optional[str] = "fr_core_news_md",
        embedding_model: Optional[str] = "all-MiniLM-L6-v2"
    ):
        self.language_model = language_model
        self.embedding_model = embedding_model
        self._nlp = None
        self._embedder = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load(self.language_model)
        return self._nlp

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def clean_text(self, text: str) -> str:
        """
        Nettoyage basique du texte : suppression des retours, tabulations et espaces multiples.
        """
        if not text:
            return ""
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def detect_language(self, text: str) -> str:
        """
        Détecte la langue principale du texte. Retourne "unknown" si erreur ou texte trop court.
        """
        text = self.clean_text(text)
        if len(text.split()) < 5:
            return "unknown"
        try:
            return detect(text)
        except:
            return "unknown"

    def extract_entities(self, text: str) -> Dict[str, list]:
        """
        Extrait les entités nommées (PERSON, ORG, GPE, etc.) en utilisant spaCy.
        Retourne un dictionnaire label -> liste d'entités (sans doublons).
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            label = ent.label_.upper()
            entities.setdefault(label, set()).add(ent.text.strip())
        return {label: sorted(list(values)) for label, values in entities.items()}

    def get_embedding(self, text: str) -> list:
        """
        Retourne l'embedding vectoriel du texte nettoyé sous forme de liste.
        """
        text = self.clean_text(text)
        return self.embedder.encode(text).tolist()

    def compute_quality(self, text: str) -> float:
        """
        Calcule un score de qualité basé sur la diversité lexicale et la densité de ponctuation.
        """
        if not text:
            return 0.0

        words = text.split()
        num_words = len(words)
        if num_words == 0:
            return 0.0

        num_unique = len(set(words))
        num_punct = sum(1 for c in text if c in ".!?")

        lexical_diversity = num_unique / num_words
        punct_ratio = num_punct / len(text)

        return round(0.4 * lexical_diversity + 0.6 * punct_ratio, 3)

    def enrich(self, article: Dict[str, Union[str, int]]) -> Dict:
        """
        Enrichit un article (dict) avec texte nettoyé, langue, entités, embedding, qualité.
        """
        text = article.get("text", "")
        cleaned = self.clean_text(text)

        enriched = {
            **article,
            "cleaned_text": cleaned,
            "language": self.detect_language(cleaned),
            "entities": self.extract_entities(cleaned),
            "embedding": self.get_embedding(cleaned),
            "quality_score": self.compute_quality(cleaned),
        }
        return enriched




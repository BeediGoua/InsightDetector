# src/models/reference_models.py

from typing import Optional, List, Dict
import spacy
import time


class LeadKModel:
    """
    Baseline : sélectionne les K premières phrases du texte.
    Utilisée comme référence rapide pour les articles d'actualité.
    """

    def __init__(self, k: int = 3, spacy_model: str = "fr_core_news_md"):
        self.k = k
        self.model_name = "lead_k"
        self.nlp = spacy.load(spacy_model)

    def summarize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return ""
        
        lead = sentences[:self.k]
        result = " ".join(sent.text.strip() for sent in lead)
        return result if result.strip() else ""
    
    def as_summary_dict(self, text: str) -> Dict:
        """
        Retourne le résumé sous forme de dictionnaire uniformisé.
        """
        start_time = time.time()
        summary = self.summarize(text)
        generation_time = time.time() - start_time
        
        return {
            "model": self.model_name,
            "type": "baseline",
            "summary": summary,
            "confidence": 0.5,
            "length": len(summary.split()),
            "generation_time": round(generation_time, 3),
        }


class EntityBasedModel:
    """
    Baseline : sélectionne les phrases contenant un minimum d'entités nommées.
    Représente un extracteur basé sur l'information saillante.
    """

    def __init__(self, min_entities: int = 1, spacy_model: str = "fr_core_news_md"):
        self.min_entities = min_entities
        self.model_name = "entity_based"
        self.nlp = spacy.load(spacy_model)

    def summarize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        
        doc = self.nlp(text)
        selected = []
        for sent in doc.sents:
            if len(sent.ents) >= self.min_entities:
                sentence_text = sent.text.strip()
                if sentence_text:
                    selected.append(sentence_text)
        
        result = " ".join(selected)
        return result if result.strip() else ""
    
    def as_summary_dict(self, text: str) -> Dict:
        """
        Retourne le résumé sous forme de dictionnaire uniformisé.
        """
        start_time = time.time()
        summary = self.summarize(text)
        generation_time = time.time() - start_time
        
        return {
            "model": self.model_name,
            "type": "baseline",
            "summary": summary,
            "confidence": 0.5,
            "length": len(summary.split()),
            "generation_time": round(generation_time, 3),
        }


class HybridHeuristicModel:
    """
    Baseline hybride :
    - Commence par la première phrase (ancrage)
    - Ajoute les phrases contenant des entités nommées
    - Coupe au bout de max_sentences

    Cible les documents semi-structurés comme les rapports ou mails.
    """

    def __init__(self,
                 max_sentences: int = 5,
                 spacy_model: str = "fr_core_news_md"):
        self.max_sentences = max_sentences
        self.model_name = "hybrid_heuristic"
        self.nlp = spacy.load(spacy_model)

    def summarize(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        summary = []

        if sentences:
            first_sentence = sentences[0].text.strip()
            if first_sentence:
                summary.append(first_sentence)

        for sent in sentences[1:]:
            if any(ent for ent in sent.ents):
                sentence_text = sent.text.strip()
                if sentence_text and sentence_text not in summary:
                    summary.append(sentence_text)
            if len(summary) >= self.max_sentences:
                break

        result = " ".join(summary[:self.max_sentences])
        return result if result.strip() else ""
    
    def as_summary_dict(self, text: str) -> Dict:
        """
        Retourne le résumé sous forme de dictionnaire uniformisé.
        """
        start_time = time.time()
        summary = self.summarize(text)
        generation_time = time.time() - start_time
        
        return {
            "model": self.model_name,
            "type": "baseline",
            "summary": summary,
            "confidence": 0.5,
            "length": len(summary.split()),
            "generation_time": round(generation_time, 3),
        }

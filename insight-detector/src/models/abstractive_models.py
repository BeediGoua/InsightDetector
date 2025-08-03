# src/models/abstractive_models.py
"""
Implémentation des modèles abstractifs : BARThez, T5-French
avec optimisations pour performance, robustesse et fallback extractif.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5Tokenizer, T5ForConditionalGeneration
)
from typing import Dict, Optional, List
import logging
import gc
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AbstractiveModelBase:
    """Classe de base pour tous les modèles abstractifs"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.generation_config = self._get_default_config()
        logger.info(f"Initialisation {self.model_name} sur {self.device.upper()}")

    def _get_default_config(self) -> Dict:
        return {
            "max_length": 150,
            "min_length": 30,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "do_sample": False,
            "temperature": 1.0,
            "repetition_penalty": 1.1,
        }

    def _generic_postprocess(self, summary: str) -> str:
        """Post-processing générique pour tous les modèles"""
        if not summary:
            return ""
        summary = re.sub(r'^(résumé|summary)[:,\s]*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'<.*?>', '', summary)
        summary = re.sub(r'▁', ' ', summary)  # artefacts SentencePiece
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary

    def load_model(self) -> bool:
        raise NotImplementedError

    def summarize(self, text: str, **kwargs) -> Dict:
        raise NotImplementedError

    def cleanup(self):
        """Nettoyage mémoire"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            logger.warning(f"Erreur lors du cleanup: {e}")
        self.is_loaded = False


class BARThezModel(AbstractiveModelBase):
    """Modèle BARThez optimisé pour résumé français"""

    def __init__(self, device: str = "cpu"):
        super().__init__("moussaKam/barthez", device)
        self.fallback_models = [
            "facebook/mbart-large-cc25",
            "facebook/bart-large-cnn",
        ]

    def load_model(self) -> bool:
        for model_name in [self.model_name] + self.fallback_models:
            try:
                logger.info(f" Chargement {model_name} sur {self.device.upper()}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.model_name = model_name
                logger.info(f" {model_name} chargé avec succès")
                return True
            except Exception as e:
                logger.warning(f" Échec {model_name}: {e}")
                self.cleanup()
                continue
        logger.error(" Aucun modèle BARThez/MBART n'a pu être chargé")
        return False

    def summarize(self, text: str, **kwargs) -> Optional[Dict]:
        if not self.is_loaded:
            return None
        start_time = time.time()
        try:
            config = {**self.generation_config, **kwargs}
            inputs = self.tokenizer(
                text.strip(),
                max_length=1024,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                summary_ids = self.model.generate(inputs["input_ids"], **config)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary_clean = self._generic_postprocess(summary)
            generation_time = time.time() - start_time
            return {
                "model": "barthez",
                "type": "abstractive_french",
                "summary": summary_clean,
                "confidence": 0.8,
                "length": len(summary_clean.split()),
                "generation_time": generation_time,
                "model_variant": self.model_name,
            }
        except Exception as e:
            logger.error(f"Erreur génération BARThez: {e}")
            return None


# mT5 Model supprimé pour améliorer la qualité des résumés


class FrenchT5Model(AbstractiveModelBase):
    """Modèle T5 spécialisé français"""

    def __init__(self, device: str = "cpu"):
        super().__init__("plguillou/t5-base-fr-sum-cnndm", device)
        self.fallback_models = ["google/t5-v1_1-small", "google/flan-t5-small"]

    def load_model(self) -> bool:
        for model_name in [self.model_name] + self.fallback_models:
            try:
                logger.info(f" Chargement T5 français: {model_name}...")
                self.tokenizer = T5Tokenizer.from_pretrained(model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                self.model_name = model_name
                logger.info(f" T5 français chargé: {model_name}")
                return True
            except Exception as e:
                logger.warning(f" Échec {model_name}: {e}")
                self.cleanup()
                continue
        return False

    def summarize(self, text: str, **kwargs) -> Optional[Dict]:
        if not self.is_loaded:
            return None
        start_time = time.time()
        try:
            config = {**self.generation_config, **kwargs}
            prompt = f"résume: {text}"
            inputs = self.tokenizer(prompt, max_length=768, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                summary_ids = self.model.generate(inputs["input_ids"], **config)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary_clean = self._generic_postprocess(summary)
            generation_time = time.time() - start_time
            return {
                "model": "french_t5",
                "type": "abstractive_french_specialized",
                "summary": summary_clean,
                "confidence": 0.8,
                "length": len(summary_clean.split()),
                "generation_time": generation_time,
                "model_variant": self.model_name,
            }
        except Exception as e:
            logger.error(f"Erreur T5 français: {e}")
            return None


class AbstractiveEnsemble:
    """Gestionnaire ensemble modèles abstractifs"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models = {}
        self.model_priorities = ["barthez", "french_t5"]  # mT5 désactivé temporairement

    def load_models(self, models_to_load: List[str] = None, max_models: int = None) -> Dict[str, bool]:
        if models_to_load is None:
            models_to_load = self.model_priorities
        results = {}
        for i, model_name in enumerate(models_to_load):
            if max_models and i >= max_models:
                break
            try:
                if model_name == "barthez":
                    model = BARThezModel(self.device)
                # mT5 supprimé définitivement
                elif model_name == "french_t5":
                    model = FrenchT5Model(self.device)
                else:
                    logger.warning(f"Modèle inconnu: {model_name}")
                    results[model_name] = False
                    continue
                success = model.load_model()
                if success:
                    self.models[model_name] = model
                results[model_name] = success
            except Exception as e:
                logger.error(f"Erreur chargement {model_name}: {e}")
                results[model_name] = False
        logger.info(f" Modèles chargés: {list(self.models.keys())}")
        return results

    def generate_all_summaries(self, text: str, **kwargs) -> List[Dict]:
        """Génération avec tous les modèles disponibles + fallback si aucun ne fonctionne."""
        summaries = []

        for model_name, model in self.models.items():
            try:
                result = model.summarize(text, **kwargs)
                if result:
                    summaries.append(result)
            except Exception as e:
                logger.error(f"Erreur génération {model_name}: {e}")

        if not summaries:
            logger.warning(" Aucun modèle n’a généré de résumé. Fallback extractif utilisé.")
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            fallback_summary = ". ".join(sentences[:3]) if sentences else "Résumé indisponible."
            summaries.append({
                "model": "fallback_extract",
                "type": "extractive_baseline",
                "summary": fallback_summary,
                "confidence": 0.4,
                "length": len(fallback_summary.split()),
                "generation_time": 0.01,
                "model_variant": "simple_baseline",
            })

        return summaries

    def cleanup_all(self):
        for model in self.models.values():
            model.cleanup()
        self.models.clear()

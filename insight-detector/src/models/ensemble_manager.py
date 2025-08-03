# src/models/ensemble_manager.py

from typing import List, Dict, Optional
import numpy as np

DOMAIN_WEIGHTS = {
    "news": {"barthez": 0.4, "camembert": 0.3, "mt5": 0.3},
    "scientific": {"camembert": 0.5, "mt5": 0.3, "barthez": 0.2},
    "legal": {"mt5": 0.5, "barthez": 0.3, "camembert": 0.2},
}


class SummaryEnsembleManager:
    """
    Gère la fusion de résumés générés par plusieurs modèles selon différentes stratégies :
    - "confidence_weighted" : pondération par score de qualité
    - "domain_based" : pondération selon le domaine (news, scientific, legal)
    - "length_adaptive" : stratégie dynamique en fonction de la longueur du texte source
    """

    def __init__(self, domain: Optional[str] = None, source_length: Optional[int] = None):
        self.domain = domain
        self.source_length = source_length

    def generate_ensemble_summary(self, summaries: List[Dict], strategy: str = "confidence_weighted") -> str:
        """
        Génère un résumé final à partir de plusieurs candidats.

        Args:
            summaries (List[Dict]): [{"model": str, "summary": str, "confidence": float}]
            strategy (str): stratégie utilisée pour la fusion

        Returns:
            str: résumé final fusionné
        """
        if not summaries:
            return self._fallback_summary(summaries)

        # Filtrer les résumés vides ou invalides
        valid_summaries = [s for s in summaries if s and (s.get("summary") or s.get("text"))]
        if not valid_summaries:
            return self._fallback_summary(summaries)

        self._normalize_summaries(valid_summaries)

        try:
            if strategy == "confidence_weighted":
                weights = self._get_confidence_weights(valid_summaries)
            elif strategy == "domain_based":
                weights = self._get_domain_weights(valid_summaries)
            elif strategy == "length_adaptive":
                weights = self._get_adaptive_weights(valid_summaries)
            else:
                raise ValueError(f"Stratégie inconnue : {strategy}")

            # Garde-fou final : vérifier que les poids sont valides
            if not weights or sum(weights) == 0 or all(w == 0 for w in weights):
                return self._fallback_summary(valid_summaries)

            return self._weighted_concat(valid_summaries, weights)
            
        except Exception as e:
            # En cas d'erreur dans la pondération, utiliser le fallback
            return self._fallback_summary(valid_summaries)

    def _normalize_summaries(self, summaries: List[Dict]) -> None:
        """
        Normalise les clés pour assurer cohérence entre les différents outputs.
        Priorise la clé 'summary' sur 'text' et 'confidence' sur 'score'.
        """
        for s in summaries:
            # Gestion flexible des clés de texte (priorise 'summary')
            text = s.get("summary") or s.get("text", "")
            if not text or not text.strip():
                text = "Résumé indisponible."
            s["text"] = text
            
            # Gestion flexible des clés de score (priorise 'confidence')
            score = s.get("confidence", s.get("score", 0))
            if score is None or score < 0:
                score = 0.5  # valeur par défaut
            s["score"] = score

    def _get_confidence_weights(self, summaries: List[Dict]) -> List[float]:
        scores = [s.get("score", 0) for s in summaries]
        total = sum(scores)
        
        # Garde-fou : si tous les scores sont 0, retourner des poids uniformes
        if total <= 0:
            return [1.0 / len(scores) for _ in scores]
        
        weights = [s / total for s in scores]
        
        # Garde-fou : s'assurer qu'au moins un poids est > 0
        if all(w == 0 for w in weights):
            return [1.0 / len(scores) for _ in scores]
        
        return weights

    def _get_domain_weights(self, summaries: List[Dict]) -> List[float]:
        weights_map = DOMAIN_WEIGHTS.get(self.domain, {})
        weights = [weights_map.get(s.get("model", "unknown"), 0) for s in summaries]
        total = sum(weights)
        
        # Garde-fou : si aucun poids de domaine ou total = 0, fallback sur confidence
        if total <= 0:
            return self._get_confidence_weights(summaries)
        
        normalized_weights = [w / total for w in weights]
        
        # Garde-fou final : s'assurer qu'au moins un poids est > 0
        if all(w == 0 for w in normalized_weights):
            return self._get_confidence_weights(summaries)
        
        return normalized_weights

    def _get_adaptive_weights(self, summaries: List[Dict]) -> List[float]:
        if self.source_length is None:
            return self._get_confidence_weights(summaries)
        elif self.source_length < 200:
            return self._get_domain_weights(summaries)
        elif self.source_length < 1000:
            return self._get_confidence_weights(summaries)
        else:
            best = max(summaries, key=lambda s: s.get("score", 0))
            return [1.0 if s == best else 0.0 for s in summaries]

    def _weighted_concat(self, summaries: List[Dict], weights: List[float]) -> str:
        """
        Combine les résumés proportionnellement aux poids associés.
        """
        ordered = sorted(zip(summaries, weights), key=lambda x: -x[1])
        combined = ""
        for summary, weight in ordered:
            repeat = int(round(weight * 10))
            combined += (" " + summary["text"]) * max(repeat, 1)
        return combined.strip()

    def _fallback_summary(self, summaries: List[Dict]) -> str:
        """
        Fallback : retourne le résumé avec le meilleur score.
        Utilisé quand la pondération échoue ou si summaries est vide.
        """
        if not summaries:
            return "Aucun résumé disponible."
        
        # Filtrer les résumés avec du contenu valide
        valid_summaries = []
        for s in summaries:
            text = s.get("summary") or s.get("text", "")
            if text and text.strip() and text.strip() != "Résumé indisponible.":
                valid_summaries.append(s)
        
        if not valid_summaries:
            return "Aucun résumé valide disponible."
        
        # Choisir le meilleur résumé basé sur le score/confidence
        best_summary = max(valid_summaries, key=lambda s: s.get("confidence", s.get("score", 0)))
        
        # Retourner le texte du meilleur résumé
        return best_summary.get("summary") or best_summary.get("text", "Résumé de fallback indisponible.")

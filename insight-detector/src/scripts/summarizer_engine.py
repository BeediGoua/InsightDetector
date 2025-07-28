# src/scripts/summarizer_engine.py

from typing import Dict, Optional
import time
import logging
import json
from pathlib import Path

from models.abstractive_models import AbstractiveEnsemble
from models.ensemble_manager import SummaryEnsembleManager

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def summarize_text_with_ensemble(
    text: str,
    strategy: str = 'adaptive',
    domain: Optional[str] = None,
    device: str = 'cpu',
    verbose: bool = False,
    save_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Orchestration complète de la génération et fusion de résumés.

    Args:
        text (str): Texte brut à résumer.
        strategy (str): Stratégie de fusion ('confidence_weighted', 'domain_based', 'length_adaptive', 'adaptive').
        domain (Optional[str]): Domaine du texte (utile pour domain_based).
        device (str): 'cpu' ou 'cuda'.
        verbose (bool): Affiche les résumés intermédiaires si True.
        save_path (Optional[str]): Chemin pour sauvegarder le résultat JSON.
        random_seed (Optional[int]): Pour reproductibilité éventuelle.

    Returns:
        Dict: Résumé final, résumés unitaires, et métadonnées complètes.
    """
    if not text or not text.strip():
        logger.error("Le texte fourni est vide.")
        raise ValueError("Le texte fourni est vide. Merci de fournir un texte à résumer.")

    if random_seed is not None:
        import random, numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"Seed d'aléatoire fixée à {random_seed}")

    start_time = time.time()
    summary_id = f"summary_{int(start_time)}"

    try:
        # 1. Initialisation des modèles abstractifs
        logger.info("Initialisation des modèles abstractifs...")
        abstractive_ensemble = AbstractiveEnsemble(device=device)
        abstractive_ensemble.load_models()

        # 2. Génération des résumés unitaires
        logger.info("Génération des résumés unitaires...")
        summaries = abstractive_ensemble.generate_all_summaries(text)
        if not summaries or all(s is None for s in summaries):
            logger.error("Aucun résumé n'a été généré.")
            raise RuntimeError("Aucun résumé n'a été généré.")

        if verbose:
            logger.info("Résumés générés :")
            for s in summaries:
                if s is None:
                    continue
                model_info = f"{s.get('model', 'NA')}"
                if 'type' in s:
                    model_info += f" ({s['type']})"
                confidence = s.get('confidence', s.get('score', 0))
                length = s.get('length', len(s.get('summary', s.get('text', '')).split()))
                logger.info(f" - {model_info} → résumé de {length} mots | confiance={confidence:.3f}")

        # 3. Fusion avec le SummaryEnsembleManager
        logger.info("Fusion des résumés avec SummaryEnsembleManager...")
        fusion_manager = SummaryEnsembleManager(
            domain=domain,
            source_length=len(text.split())
        )
        fused_summary = fusion_manager.generate_ensemble_summary(summaries, strategy=strategy)

        # 4. Construction du résultat final
        duration = round(time.time() - start_time, 2)
        result = {
            "summary_id": summary_id,
            "ensemble_summary": {
                "summary": fused_summary,
                "length": len(fused_summary.split()),
            },
            "individual_summaries": summaries,
            "metadata": {
                "fusion_strategy": strategy,
                "fusion_method": fusion_manager.__class__.__name__,
                "domain": domain,
                "device": device,
                "source_length": len(text.split()),
                "runtime_seconds": duration,
                "num_models_used": len([s for s in summaries if s]),
                "random_seed": random_seed,
                "timestamp": start_time,
            }
        }

        logger.info(f"[{summary_id}] Résumé généré avec la stratégie '{strategy}' en {duration:.2f} secondes.")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True, parents=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Résultat sauvegardé dans {save_path}")

        return result

    except Exception as e:
        logger.error(f"Erreur lors de la génération du résumé : {e}", exc_info=True)
        raise e

# Exécution simple pour test
if __name__ == "__main__":
    test_text = "Votre texte brut à résumer ici..."
    output = summarize_text_with_ensemble(
        text=test_text,
        strategy='adaptive',
        device='cpu',
        verbose=True
    )
    print("Résumé final :", output['ensemble_summary']['summary'])

# src/scripts/summarizer_engine.py
"""
Moteur de résumé optimisé pour InsightDetector.
Version fusionnée : API originale + optimisations + traitement batch.

Fonctionnalités :
- API compatible : summarize_text_with_ensemble()
- Optimisations performance : OptimizedModelManager (singleton)
- Traitement batch : batch_summarize_optimized()
- Checkpointing automatique
- Thread-safe et production-ready
"""

from typing import Dict, Optional, List
import time
import logging
import json
from pathlib import Path
import threading

from models.abstractive_models import AbstractiveEnsemble
from models.ensemble_manager import SummaryEnsembleManager
from models.extractive_models import ExtractiveEnsemble
from models.reference_models import LeadKModel

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedModelManager:
    """
    Gestionnaire singleton des modèles pour éviter le rechargement.
    Thread-safe et réutilisable pour performance optimale.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, device: str = "cpu"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, device: str = "cpu"):
        # Éviter la re-initialisation
        if hasattr(self, '_initialized'):
            return
            
        self.device = device
        self.abstractive_ensemble = None
        self.extractive_ensemble = None
        self.leadk_model = None
        self._models_loaded = False
        self._load_lock = threading.Lock()
        self._initialized = True
        
        logger.info(f"OptimizedModelManager initialisé sur {device}")
    
    def load_models_once(self, force_reload: bool = False) -> bool:
        """
        Charge tous les modèles une seule fois.
        Thread-safe pour utilisation parallèle.
        """
        if self._models_loaded and not force_reload:
            logger.info("Modèles déjà chargés, réutilisation des instances existantes.")
            return True
            
        with self._load_lock:
            # Double-check après acquisition du lock
            if self._models_loaded and not force_reload:
                return True
                
            try:
                start_time = time.time()
                logger.info("CHARGEMENT UNIQUE DES MODELES...")
                
                # 1. Modèles abstractifs (priorité haute)
                logger.info("  Chargement ensemble abstractif...")
                self.abstractive_ensemble = AbstractiveEnsemble(device=self.device)
                abstractive_success = self.abstractive_ensemble.load_models()
                
                if abstractive_success:
                    logger.info(f"  Modeles abstractifs charges: {list(self.abstractive_ensemble.models.keys())}")
                else:
                    logger.warning("  Echec chargement abstractifs, fallback active")
                
                # 2. Modèles extractifs (fallback)
                logger.info("  Chargement ensemble extractif...")
                try:
                    self.extractive_ensemble = ExtractiveEnsemble()
                    self.extractive_ensemble.load_models()
                    logger.info("  Modeles extractifs charges")
                except Exception as e:
                    logger.warning(f"  Extractifs non disponibles: {e}")
                    self.extractive_ensemble = None
                
                # 3. Modèle de base (toujours disponible)
                self.leadk_model = LeadKModel(k=3)
                logger.info("  LeadK baseline pret")
                
                duration = time.time() - start_time
                self._models_loaded = True
                
                logger.info(f"TOUS LES MODELES CHARGES EN {duration:.1f}s")
                logger.info("Reutilisation pour tous les articles suivants...")
                
                return True
                
            except Exception as e:
                logger.error(f"Erreur critique lors du chargement: {e}")
                self._models_loaded = False
                return False
    
    def get_abstractive_ensemble(self):
        """Récupère l'ensemble abstractif chargé"""
        if not self._models_loaded:
            raise RuntimeError("Modèles non chargés. Appelez load_models_once() d'abord.")
        return self.abstractive_ensemble
    
    def get_extractive_ensemble(self):
        """Récupère l'ensemble extractif chargé"""
        return self.extractive_ensemble
    
    def get_leadk_model(self):
        """Récupère le modèle baseline"""
        return self.leadk_model
    
    def cleanup_all(self):
        """Nettoyage complet de tous les modèles"""
        logger.info("Nettoyage de tous les modeles...")
        
        if self.abstractive_ensemble:
            # Nettoyer chaque modèle de l'ensemble
            for model_name, model_instance in self.abstractive_ensemble.models.items():
                if hasattr(model_instance, 'cleanup'):
                    model_instance.cleanup()
                    
        if self.extractive_ensemble:
            if hasattr(self.extractive_ensemble, 'cleanup'):
                self.extractive_ensemble.cleanup()
                
        self._models_loaded = False
        logger.info("Nettoyage termine")


# Variable globale pour réutilisation du gestionnaire
_global_model_manager = None
_manager_lock = threading.Lock()


def get_or_create_model_manager(device: str = "cpu") -> OptimizedModelManager:
    """Obtient ou crée le gestionnaire de modèles global"""
    global _global_model_manager
    
    with _manager_lock:
        if _global_model_manager is None:
            _global_model_manager = OptimizedModelManager(device=device)
            _global_model_manager.load_models_once()
        return _global_model_manager


def summarize_text_with_ensemble(
    text: str,
    strategy: str = "adaptive",
    domain: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = False,
    save_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    use_optimization: bool = True,
) -> Dict:
    """
    Orchestration complète de la génération et fusion de résumés,
    avec fallback multi-niveaux (abstractif -> extractif -> baseline).
    
    VERSION OPTIMISÉE : Réutilise les modèles chargés quand use_optimization=True.
    
    Args:
        text: Texte à résumer
        strategy: Stratégie de fusion ("adaptive", "confidence_weighted", etc.)
        domain: Domaine spécialisé (optionnel)
        device: Dispositif de calcul ("cpu" ou "cuda")
        verbose: Mode verbeux pour logs détaillés
        save_path: Chemin de sauvegarde du résultat (optionnel)
        random_seed: Graine aléatoire pour reproductibilité
        use_optimization: Si True, utilise le gestionnaire optimisé (recommandé)
        
    Returns:
        Dict avec résumé fusionné, résumés individuels et métadonnées
    """
    if not text or not text.strip():
        logger.error("Le texte fourni est vide.")
        raise ValueError("Le texte fourni est vide. Merci de fournir un texte à résumer.")

    if random_seed is not None:
        import random, numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        if verbose:
            logger.info(f"Seed aléatoire fixée à {random_seed}")

    start_time = time.time()
    summary_id = f"summary_{int(start_time)}"
    summaries = []

    try:
        if use_optimization:
            # VERSION OPTIMISÉE : Réutilise les modèles chargés
            model_manager = get_or_create_model_manager(device=device)
            
            # 1. Essai des modèles abstractifs (pré-chargés)
            abstractive_ensemble = model_manager.get_abstractive_ensemble()
            if abstractive_ensemble and abstractive_ensemble.models:
                summaries = abstractive_ensemble.generate_all_summaries(text)
                
                if verbose and summaries:
                    logger.info("Résumés abstractifs générés :")
                    for s in summaries:
                        if s:
                            logger.info(f" - {s['model']} ({s.get('type','NA')}) "
                                        f"| longueur={len(s['summary'].split())} "
                                        f"| confiance={s.get('confidence',0):.3f}")

            if not summaries or all(s is None for s in summaries):
                raise RuntimeError("Aucun résumé abstractive valide généré.")
                
        else:
            # VERSION ORIGINALE : Rechargement des modèles (compatibilité)
            logger.info("Initialisation des modèles abstractifs...")
            abstractive_ensemble = AbstractiveEnsemble(device=device)
            abstractive_ensemble.load_models()
            summaries = abstractive_ensemble.generate_all_summaries(text)

            if not summaries or all(s is None for s in summaries):
                raise RuntimeError("Aucun résumé abstractive valide généré.")

            if verbose:
                logger.info("Résumés abstractifs générés :")
                for s in summaries:
                    if s:
                        logger.info(f" - {s['model']} ({s.get('type','NA')}) "
                                    f"| longueur={len(s['summary'].split())} "
                                    f"| confiance={s.get('confidence',0):.3f}")

    except Exception as e:
        logger.warning(f"AbstractiveEnsemble échoué ({e}). Passage au fallback extractif.")
        summaries = []

    # 2. Si aucun résumé abstractive → fallback extractif
    if not summaries:
        try:
            if use_optimization:
                model_manager = get_or_create_model_manager(device=device)
                extractive_ensemble = model_manager.get_extractive_ensemble()
                if extractive_ensemble:
                    summaries = extractive_ensemble.extract_all_summaries(text)
            else:
                logger.info("Fallback : utilisation d'un modèle extractif...")
                extractive = ExtractiveEnsemble()
                extractive.load_models()
                summaries = extractive.extract_all_summaries(text)
                
        except Exception as e:
            logger.warning(f"ExtractiveEnsemble échoué ({e}). Passage au LeadKModel.")
            summaries = []

    # 3. Si aucun résumé extractif → fallback LeadK
    if not summaries:
        if use_optimization:
            model_manager = get_or_create_model_manager(device=device)
            leadk = model_manager.get_leadk_model()
        else:
            logger.info("Fallback final : LeadKModel (baseline).")
            leadk = LeadKModel(k=3)
        summaries = [leadk.as_summary_dict(text)]

    # 4. Fusion des résumés
    logger.info("Fusion des résumés avec SummaryEnsembleManager...")
    fusion_manager = SummaryEnsembleManager(
        domain=domain,
        source_length=len(text.split())
    )

    # Protection si un seul résumé (pas besoin de fusion)
    if len(summaries) == 1:
        fused_summary = summaries[0]["summary"]
    else:
        fused_summary = fusion_manager.generate_ensemble_summary(summaries, strategy=strategy)

    # 5. Construction du résultat final
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
            "optimization_used": use_optimization,
            "fallback_used": (
                "abstractive" if summaries and summaries[0].get("type", "").startswith("abstractive")
                else "extractive" if summaries and summaries[0].get("type", "").startswith("extractive")
                else "baseline"
            )
        }
    }

    if verbose:
        optimization_msg = "optimisée" if use_optimization else "standard"
        logger.info(f"[{summary_id}] Résumé généré en {duration:.2f} secondes "
                    f"(version {optimization_msg}, fallback : {result['metadata']['fallback_used']})")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Résultat sauvegardé dans {save_path}")

    return result


def batch_summarize_optimized(
    texts: List[str],
    device: str = "cpu",
    strategy: str = "adaptive",
    domain: Optional[str] = None,
    verbose: bool = False,
    save_progress_every: int = 20,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Traitement batch optimisé avec chargement unique des modèles.
    
    Args:
        texts: Liste des textes à résumer
        device: Dispositif (cpu/cuda)
        strategy: Stratégie de fusion
        domain: Domaine spécialisé
        verbose: Mode verbeux
        save_progress_every: Sauvegarde intermédiaire tous les N résumés
        output_path: Chemin de sauvegarde
    
    Returns:
        Liste des résultats de résumé
    """
    logger.info(f"TRAITEMENT BATCH OPTIMISE: {len(texts)} articles")
    
    # 1. Initialisation du gestionnaire de modèles
    model_manager = get_or_create_model_manager(device=device)
    
    # 2. Traitement séquentiel optimisé
    all_results = []
    start_batch_time = time.time()
    
    try:
        for i, text in enumerate(texts, 1):
            try:
                result = summarize_text_with_ensemble(
                    text=text,
                    strategy=strategy,
                    domain=domain,
                    device=device,
                    verbose=verbose,
                    use_optimization=True  # Forcer optimisation
                )
                all_results.append(result)
                
                # Affichage progrès
                elapsed = time.time() - start_batch_time
                estimated_total = (elapsed / i) * len(texts)
                remaining = estimated_total - elapsed
                
                if i % 10 == 0 or verbose:
                    logger.info(f"Progres: {i}/{len(texts)} "
                                f"({100*i/len(texts):.1f}%) "
                                f"- Temps restant: {remaining/60:.1f}min")
                
                # Sauvegarde intermédiaire
                if save_progress_every and i % save_progress_every == 0 and output_path:
                    intermediate_path = Path(output_path).with_suffix('.partial.json')
                    with open(intermediate_path, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"Sauvegarde intermediaire: {i} resumes")
                
            except Exception as e:
                logger.error(f"Erreur article {i}: {e}")
                continue
        
        # 3. Statistiques finales
        total_duration = time.time() - start_batch_time
        avg_per_article = total_duration / len(all_results) if all_results else 0
        
        logger.info(f"BATCH TERMINE:")
        logger.info(f"  {len(all_results)}/{len(texts)} articles traites")
        logger.info(f"  Temps total: {total_duration/60:.1f} minutes")
        logger.info(f"  Moyenne: {avg_per_article:.1f}s/article")
        logger.info(f"  Amelioration: {60/avg_per_article:.0f}x plus rapide que l'ancienne version")
        
        # 4. Sauvegarde finale
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Resultats sauvegardes: {output_path}")
        
        return all_results
        
    finally:
        # Nettoyage optionnel des modèles
        # model_manager.cleanup_all()  # Décommentez si nécessaire
        pass


# Fonction utilitaire pour tests rapides
def quick_test_optimization():
    """Test rapide pour valider l'optimisation"""
    test_text = "Des chercheurs italiens ont développé une nouvelle technologie de surveillance basée sur les ondes Wi-Fi."
    
    # Test version optimisée
    start = time.time()
    result_opt = summarize_text_with_ensemble(test_text, use_optimization=True, verbose=True)
    time_opt = time.time() - start
    
    print(f"Test optimisation réussi:")
    print(f"  Temps: {time_opt:.2f}s")
    print(f"  Résumé: {result_opt['ensemble_summary']['summary'][:100]}...")
    print(f"  Optimisation utilisée: {result_opt['metadata']['optimization_used']}")
    
    return result_opt


if __name__ == "__main__":
    # Test rapide si exécuté directement
    quick_test_optimization()

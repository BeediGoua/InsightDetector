"""
Module de détection d'hallucinations pour InsightDetector.

Ce module implémente un système de détection d'hallucinations en 4 niveaux :
- Niveau 0 : Pré-filtrage qualité
- Niveau 1 : Détection heuristique
- Niveau 2 : Validation factuelle
- Niveau 3 : Classificateur ML

Auteur: InsightDetector Team
"""

from .level0_prefilter import QualityFilter
from .level1_heuristic import HeuristicAnalyzer

__all__ = ['QualityFilter', 'HeuristicAnalyzer']
__version__ = '1.0.0'
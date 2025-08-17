"""
Niveau 2 : Validation factuelle approfondie (<1s)

Module de validation factuelle adapté aux résultats du Niveau 1 :
- 325 candidats fact-check sur 171 résumés
- 183 suspects avec coherence <0.7
- Validation multi-tiers selon complexité

Auteur: InsightDetector Team
Version: 2.0.0 - Optimisé pour données réelles
"""

from .level2_coordinator import Level2FactualProcessor
from .coherence_factuality_validator import CoherenceFactualityValidator
from .candidate_validator import CandidateValidator
from .statistical_fact_validator import StatisticalFactValidator
from .internal_consistency_analyzer import InternalConsistencyAnalyzer

__all__ = [
    'Level2FactualProcessor',
    'CoherenceFactualityValidator', 
    'CandidateValidator',
    'StatisticalFactValidator',
    'InternalConsistencyAnalyzer'
]

__version__ = '2.0.0'
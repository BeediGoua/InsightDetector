# src/detection/level3_improvement/__init__.py
"""
Level 3 Improvement System - Amélioration intelligente des cas CRITICAL
Transforme les summaries avec factuality correcte mais coherence défaillante
"""

from .level3_processor import Level3Processor
from .text_improver import TextImprover  
from .fact_validator import FactValidator
from .config import Level3Config

__all__ = ['Level3Processor', 'TextImprover', 'FactValidator', 'Level3Config']
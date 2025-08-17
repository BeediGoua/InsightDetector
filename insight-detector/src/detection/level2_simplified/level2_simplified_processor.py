

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# --- helpers for source_id ------------------------------------------------
from hashlib import sha1
import re

def _normalize(s: str) -> str:
    """Nettoyage simple et déterministe pour le calcul de hash."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def make_source_id(url: str = None,
                   title: str = None,
                   published: str = None,
                   source: str = None) -> str:
    """
    Stratégie simple et stable:
    - Si URL disponible: source_id = SHA1(normalize(url))
    - Sinon: source_id = SHA1(normalize(source) + '|' + normalize(title) + '|' + normalize(published))
    """
    base = url or f"{_normalize(source)}|{_normalize(title)}|{_normalize(published)}"
    return sha1(_normalize(base).encode("utf-8")).hexdigest()

def resolve_source_id(summary_data: Dict[str, Any]) -> Optional[str]:
    """
    Résout un `source_id` à partir d'un item `summary_data` de Level-2.
    - Priorité: summary_data['source_id'] ou summary_data['metadata']['source_id'] si déjà présent
    - Sinon, tente de le fabriquer depuis metadata.{source_url/url, source_title/title, published, source}
    - Retourne None si aucune métadonnée exploitable n'est disponible
    """
    # 1) Déjà présent ?
    existing = summary_data.get("source_id")
    if not existing:
        meta = summary_data.get("metadata") or {}
        existing = meta.get("source_id")
    if existing:
        return existing

    # 2) Fabriquer depuis metadata si possible
    meta = summary_data.get("metadata") or {}
    url = meta.get("source_url") or meta.get("url")
    title = meta.get("source_title") or meta.get("title")
    published = meta.get("published")
    source = meta.get("source")
    if url or title or published or source:
        return make_source_id(url=url, title=title, published=published, source=source)

    # 3) Pas de métadonnée exploitable
    return None
# -------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class TierClassification(Enum):
    """Classification intelligente basée sur analyse Level 1"""
    EXCELLENT = "EXCELLENT"      # A+/A + coherence haute → Validation minimale
    GOOD = "GOOD"                # A/B+ + coherence OK → Validation ciblée
    MODERATE = "MODERATE"        # B+/B + issues → Validation approfondie
    CRITICAL = "CRITICAL"        # C/D ou issues critiques → Validation exhaustive


@dataclass
class Level2SimplifiedResult:
    """Résultat simplifié mais complet"""
    summary_id: str

    # Classification et validation
    tier_classification: TierClassification
    is_valid: bool
    validation_confidence: float

    # Métriques héritées Level 1
    grade_score: float
    coherence_score: float
    factuality_score: float
    issues_count: int

    # Priorisation Level 3
    level3_priority: float
    level3_justification: str

    # Performance
    processing_time_ms: float

    # Détails pour audit
    decision_factors: Dict[str, Any]


class Level2SimplifiedProcessor:
    """
    Processeur Level 2 Simplifié

    Philosophie: Exploiter au maximum les insights Level 1 validés
    au lieu de refaire des analyses complexes potentiellement bugguées.
    Ajout: écriture d'un `source_id` directement dans les dicts de sortie
    (pour une traçabilité parfaite au Level-3).
    """

    def __init__(self, mode: str = "balanced"):
        self.mode = mode
        self.stats = {
            'processed': 0,
            'validated': 0,
            'rejected': 0,
            'by_tier': {tier.value: 0 for tier in TierClassification}
        }

        # Seuils calibrés sur les données Level 1 réelles
        self._configure_thresholds()

        logger.info(f"Level 2 Simplifié initialisé en mode {mode}")

    def _configure_thresholds(self):
        """Configuration basée sur l'analyse Level 1 des 372 résumés - Version calibrée"""
        if self.mode == "balanced":
            self.thresholds = {
                # Seuils coherence basés sur patterns réels Level 1
                'excellent_coherence': 0.85,   # Abaissé de 0.9 → plus de EXCELLENT
                'good_coherence': 0.6,         # Abaissé de 0.7 → plus de GOOD
                'acceptable_coherence': 0.25,  # Abaissé de 0.3 → plus tolérant

                # Seuils factuality
                'high_factuality': 0.9,        # Abaissé de 0.95
                'medium_factuality': 0.7,      # Abaissé de 0.8
                'low_factuality': 0.4,         # Abaissé de 0.5

                # Seuils issues (based on Level 1 stats)
                'max_acceptable_issues': 4,    # Augmenté de 3 → plus tolérant
                'critical_issues': 10,         # Augmenté de 8 → moins restrictif

                # Budgets temps réalistes (indicatifs)
                'target_time_ms': {
                    'EXCELLENT': 2,    # Validation minimale
                    'GOOD': 5,         # Validation ciblée
                    'MODERATE': 10,    # Validation approfondie
                    'CRITICAL': 15     # Validation exhaustive
                }
            }

        logger.info(f"Seuils configurés pour mode {self.mode}")

    def classify_tier(self, summary_data: Dict[str, Any]) -> TierClassification:
        """
        Classification intelligente basée sur patterns Level 1 validés - Version calibrée

        Logique ajustée:
        - A+/A + haute cohérence → EXCELLENT (validation minimale)
        - A+/A avec issues OU B+ haute cohérence → GOOD
        - B+/B avec issues modérées → MODERATE
        - C/D OU issues critiques → CRITICAL
        - production_ready=False: CRITICAL seulement pour grades C/D
        """
        grade = summary_data.get('original_grade', 'D')
        coherence = float(summary_data.get('coherence', 0.0))
        factuality = float(summary_data.get('factuality', 0.0))
        issues_count = int(summary_data.get('num_issues', 0))
        production_ready = bool(summary_data.get('production_ready', True))

        # Classification plus nuancée si non production_ready
        if not production_ready:
            if grade in ['A+', 'A']:
                return TierClassification.GOOD
            elif grade in ['B+', 'B']:
                return TierClassification.MODERATE
            else:
                return TierClassification.CRITICAL

        # Classification standard
        if grade in ['A+', 'A']:
            if coherence >= self.thresholds['excellent_coherence'] and issues_count <= 1:
                return TierClassification.EXCELLENT
            elif coherence >= self.thresholds['good_coherence']:
                return TierClassification.GOOD
            else:
                return TierClassification.MODERATE

        elif grade in ['B+', 'B']:
            if coherence >= self.thresholds['excellent_coherence'] and issues_count <= 2:
                return TierClassification.GOOD
            elif coherence >= self.thresholds['acceptable_coherence']:
                return TierClassification.MODERATE
            else:
                return TierClassification.CRITICAL

        else:  # C, D
            return TierClassification.CRITICAL

    def calculate_validation_confidence(self, summary_data: Dict[str, Any],
                                        tier: TierClassification) -> float:
        """
        Calcul de confiance basé sur métriques Level 1 + tier - Version moins pénalisante
        """
        grade = summary_data.get('original_grade', 'D')
        coherence = float(summary_data.get('coherence', 0.0))
        factuality = float(summary_data.get('factuality', 0.0))
        issues_count = int(summary_data.get('num_issues', 0))

        # Scores de base rehaussés
        grade_scores = {'A+': 0.98, 'A': 0.95, 'B+': 0.80, 'B': 0.70, 'C': 0.45, 'D': 0.25}
        base_score = grade_scores.get(grade, 0.2)

        # Ajustements moins pénalisants
        coherence_bonus = min(0.15, (coherence - 0.3) * 0.5) if coherence > 0.3 else -0.1
        factuality_bonus = min(0.15, (factuality - 0.6) * 0.3) if factuality > 0.6 else 0.0
        issues_penalty = -0.03 * max(0, issues_count - 3)

        tier_adjustments = {
            TierClassification.EXCELLENT: 0.02,
            TierClassification.GOOD: 0.01,
            TierClassification.MODERATE: -0.05,
            TierClassification.CRITICAL: -0.15
        }

        confidence = base_score + coherence_bonus + factuality_bonus + issues_penalty + tier_adjustments[tier]
        return max(0.1, min(1.0, confidence))

    def calculate_level3_priority(self, summary_data: Dict[str, Any],
                                  confidence: float,
                                  tier: TierClassification) -> Tuple[float, str]:
        """
        Priorisation intelligente pour Level 3 ML - Version graduée
        """
        grade = summary_data.get('original_grade', 'D')
        issues_count = int(summary_data.get('num_issues', 0))
        production_ready = bool(summary_data.get('production_ready', True))

        # Priorité de base inverse de la confiance
        base_priority = 1.0 - confidence

        if not production_ready and grade in ['C', 'D']:
            priority = 1.0
            justification = f"Grade {grade} non production ready - traitement ML critique"
        elif not production_ready and grade in ['B+', 'B']:
            priority = max(0.7, base_priority)
            justification = f"Grade {grade} non production ready - traitement ML élevé"
        elif not production_ready and grade in ['A+', 'A']:
            priority = max(0.5, base_priority)
            justification = f"Grade {grade} non production ready - traitement ML standard"
        elif grade in ['C', 'D']:
            priority = max(0.8, base_priority)
            justification = f"Grade {grade} - requires ML validation"
        elif tier == TierClassification.CRITICAL:
            priority = max(0.6, base_priority)
            justification = "Tier CRITICAL - validation ML prioritaire"
        elif issues_count >= 7:
            priority = max(0.5, base_priority)
            justification = f"{issues_count} issues détectées - analyse ML recommandée"
        else:
            priority = base_priority
            justification = "Priorité standard basée sur confiance"

        return min(1.0, priority), justification

    def process_summary(self, summary_data: Dict[str, Any]) -> Level2SimplifiedResult:
        """
        Traitement simplifié mais complet d'un résumé
        """
        start_time = time.time()

        # NB: certains jeux utilisent 'id' comme identifiant de résumé, d'autres 'summary_id'
        summary_id = summary_data.get('id') or summary_data.get('summary_id') or 'unknown'

        try:
            # --- NEW: résout et injecte source_id pour L3 ----------------------
            sid = resolve_source_id(summary_data)
            if sid:
                summary_data["source_id"] = sid  # Traçabilité pour Level-3
            # ------------------------------------------------------------------

            # 1. Classification tier (logique métier pure)
            tier = self.classify_tier(summary_data)

            # 2. Calcul confiance (basé sur métriques Level 1)
            confidence = self.calculate_validation_confidence(summary_data, tier)

            # 3. Décision validation (seuils simples)
            is_valid = (
                confidence >= 0.5
                and tier != TierClassification.CRITICAL
                and bool(summary_data.get('production_ready', True))
            )

            # 4. Priorisation Level 3
            level3_priority, justification = self.calculate_level3_priority(
                summary_data, confidence, tier
            )

            # 5. Métriques héritées Level 1 (score simplifié par grade)
            grade_scores = {'A+': 1.0, 'A': 0.9, 'B+': 0.7, 'B': 0.6, 'C': 0.4, 'D': 0.2}
            grade_score = grade_scores.get(summary_data.get('original_grade', 'D'), 0.2)

            processing_time = (time.time() - start_time) * 1000.0

            # 6. Résultat structuré
            result = Level2SimplifiedResult(
                summary_id=summary_id,
                tier_classification=tier,
                is_valid=is_valid,
                validation_confidence=confidence,
                grade_score=grade_score,
                coherence_score=float(summary_data.get('coherence', 0.0)),
                factuality_score=float(summary_data.get('factuality', 0.0)),
                issues_count=int(summary_data.get('num_issues', 0)),
                level3_priority=level3_priority,
                level3_justification=justification,
                processing_time_ms=processing_time,
                decision_factors={
                    'tier_rationale': f"Grade {summary_data.get('original_grade')} + coherence {summary_data.get('coherence', 0):.3f}",
                    'validation_rationale': f"Confidence {confidence:.3f} vs threshold 0.5",
                    'production_ready': bool(summary_data.get('production_ready', True)),
                    'original_metrics': {
                        'coherence': summary_data.get('coherence'),
                        'factuality': summary_data.get('factuality'),
                        'issues': summary_data.get('detected_issues', '')
                    },
                    # NEW: trace minimale utile au debug
                    'has_source_id': bool(sid)
                }
            )

            # Stats
            self.stats['processed'] += 1
            self.stats['by_tier'][tier.value] += 1
            if is_valid:
                self.stats['validated'] += 1
            else:
                self.stats['rejected'] += 1

            return result

        except Exception as e:
            logger.error(f"Erreur traitement {summary_id}: {e}")
            # Résultat de fallback sécurisé
            return Level2SimplifiedResult(
                summary_id=summary_id,
                tier_classification=TIERClassification.CRITICAL if 'TIERClassification' in globals() else TierClassification.CRITICAL,
                is_valid=False,
                validation_confidence=0.0,
                grade_score=0.0,
                coherence_score=0.0,
                factuality_score=0.0,
                issues_count=999,
                level3_priority=1.0,
                level3_justification="Erreur de traitement - priorité maximale",
                processing_time_ms=(time.time() - start_time) * 1000.0,
                decision_factors={'error': str(e)}
            )

    def process_batch(self, summaries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Level2SimplifiedResult]]:
        """
        Traitement par batch avec gestion d'erreurs robuste
        - `valid_summaries` sont les dicts d'entrée potentiellement modifiés (incluant désormais `source_id`).
        """
        valid_summaries = []
        all_results: List[Level2SimplifiedResult] = []

        logger.info(f"Traitement Level 2 Simplifié: {len(summaries)} résumés")

        for summary_data in summaries:
            result = self.process_summary(summary_data)
            all_results.append(result)

            if result.is_valid:
                valid_summaries.append(summary_data)

        logger.info(
            f"Level 2 Simplifié terminé: {len(valid_summaries)}/{len(summaries)} validés "
            f"({(len(valid_summaries)/len(summaries)*100.0) if summaries else 0.0:.1f}%)"
        )

        return valid_summaries, all_results

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques de traitement"""
        if self.stats['processed'] > 0:
            validation_rate = self.stats['validated'] / self.stats['processed']
        else:
            validation_rate = 0.0

        return {
            'total_processed': self.stats['processed'],
            'validation_rate': validation_rate,
            'tier_distribution': self.stats['by_tier'],
            'mode': self.mode,
            'thresholds': getattr(self, 'thresholds', {})
        }

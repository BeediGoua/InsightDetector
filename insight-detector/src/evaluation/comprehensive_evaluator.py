# src/evaluation/comprehensive_evaluator.py

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Optional

from evaluation.automatic_metrics import AutomaticMetrics
from evaluation.advanced_metrics import AdvancedMetrics
from evaluation.business_metrics import BusinessMetrics
from evaluation.human_evaluation import HumanEvaluationInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """
    Évaluateur global combinant toutes les métriques (automatiques, avancées, business, humaines).
    """

    def __init__(self, device: str = 'cpu', lang: str = 'fr'):
        self.device = device
        self.lang = lang

        self.automatic_metrics = AutomaticMetrics(device, lang)
        self.advanced_metrics = AdvancedMetrics(device)
        self.business_metrics = BusinessMetrics()
        self.human_evaluator = HumanEvaluationInterface()
        logger.info("Évaluateur complet initialisé.")

    def evaluate_summary(self, summary: str, source: str, reference: Optional[str] = None) -> Dict:
        """
        Retourne toutes les métriques pour un résumé donné.
        """
        start_time = time.time()
        results = {
            'timestamp': time.time(),
            'summary': summary,
            'source_length': len(source.split()),
            'summary_length': len(summary.split())
        }

        # Automatiques (si ref dispo)
        if reference:
            results['rouge_scores'] = self.automatic_metrics.calculate_rouge_scores(summary, reference)
            results['bert_score'] = self.automatic_metrics.calculate_bert_score(summary, reference)
            results['meteor_score'] = self.automatic_metrics.calculate_meteor_score(summary, reference)

        results['abstractiveness'] = self.automatic_metrics.calculate_abstractiveness(summary, source)
        results['compression'] = self.automatic_metrics.calculate_compression_ratio(summary, source)

        # Avancées
        results['factuality'] = self.advanced_metrics.calculate_factuality_score(summary, source)
        results['hallucinations'] = self.advanced_metrics.detect_hallucinations(summary, source)
        results['coherence'] = self.advanced_metrics.calculate_coherence_score(summary)

        # Business
        results['readability'] = self.business_metrics.calculate_readability_score(summary)
        results['engagement'] = self.business_metrics.calculate_engagement_score(summary)
        results['information_density'] = self.business_metrics.calculate_information_density(summary, source)

        # Score global
        results['composite_score'] = self._calculate_composite_score(results)
        results['evaluation_time'] = time.time() - start_time

        return results

    def _calculate_composite_score(self, results: Dict) -> Dict[str, float]:
        """
        Score composite pondéré selon les métriques clés.
        """
        scores = {}
        weights = {}

        # Ajoute chaque métrique pondérée si présente
        if 'rouge_scores' in results and 'rouge1_f' in results['rouge_scores']:
            scores['rouge1'] = results['rouge_scores']['rouge1_f']
            weights['rouge1'] = 0.2
        if 'bert_score' in results and 'bertscore_f1' in results['bert_score']:
            scores['bertscore'] = results['bert_score']['bertscore_f1']
            weights['bertscore'] = 0.2
        if 'factuality' in results and 'factuality_score' in results['factuality']:
            scores['factuality'] = results['factuality']['factuality_score']
            weights['factuality'] = 0.2
        if 'readability' in results and 'readability_score' in results['readability']:
            scores['readability'] = results['readability']['readability_score']
            weights['readability'] = 0.15
        if 'engagement' in results and 'engagement_score' in results['engagement']:
            scores['engagement'] = results['engagement']['engagement_score']
            weights['engagement'] = 0.15
        if 'coherence' in results and 'coherence_score' in results['coherence']:
            scores['coherence'] = results['coherence']['coherence_score']
            weights['coherence'] = 0.1

        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            composite_score = sum(scores[k] * normalized_weights[k] for k in scores.keys())
        else:
            composite_score = 0.5

        return {
            'composite_score': composite_score,
            'component_scores': scores,
            'weights_used': weights,
            'components_count': len(scores)
        }

    def batch_evaluate(self, summaries_data: List[Dict], source: str,
                      references: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Évaluation en batch, retourne un DataFrame de tous les scores.
        """
        results = []
        for i, summary_data in enumerate(summaries_data):
            summary_text = summary_data.get('summary', '')
            model_name = summary_data.get('model', 'unknown')
            reference = references[i] if references and i < len(references) else None
            evaluation = self.evaluate_summary(summary_text, source, reference)
            flat_result = {'model': model_name, 'summary': summary_text}
            self._flatten_dict(evaluation, flat_result)
            results.append(flat_result)
        return pd.DataFrame(results)

    def _flatten_dict(self, nested_dict: Dict, flat_dict: Dict, parent_key: str = ''):
        """
        Aplatis un dictionnaire imbriqué pour DataFrame.
        """
        for key, value in nested_dict.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            if isinstance(value, dict):
                self._flatten_dict(value, flat_dict, new_key)
            elif isinstance(value, (list, tuple)):
                if value and isinstance(value[0], (int, float)):
                    flat_dict[f"{new_key}_mean"] = float(np.mean(value))
                    flat_dict[f"{new_key}_count"] = len(value)
                else:
                    flat_dict[new_key] = str(value)
            else:
                flat_dict[new_key] = value

# src/evaluation/human_evaluation.py

import numpy as np
import json
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HumanEvaluationInterface:
    """
    Interface pour la gestion des évaluations humaines :
    - création de tâches à annoter
    - chargement et analyse des résultats d’annotation humaine
    """

    def __init__(self, output_dir: str = "human_evaluations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluation_schema = self._define_schema()

    def _define_schema(self) -> Dict[str, Dict]:
        """
        Schéma des critères à annoter : fluency, adequacy, conciseness, overall
        """
        return {
            'fluency': {
                'scale': [1, 2, 3, 4, 5],
                'description': 'Qualité linguistique et naturel du résumé',
            },
            'adequacy': {
                'scale': [1, 2, 3, 4, 5],
                'description': 'Couverture du contenu important',
            },
            'conciseness': {
                'scale': [1, 2, 3, 4, 5],
                'description': 'Concision sans perte d’information',
            },
            'overall': {
                'scale': [1, 2, 3, 4, 5],
                'description': 'Qualité globale du résumé',
            }
        }

    def create_evaluation_task(self, summaries: List[Dict], source_text: str, task_id: str) -> Dict:
        """
        Crée un fichier JSON à annoter humainement.
        summaries: liste de dicos {"model", "summary"}
        """
        task = {
            'task_id': task_id,
            'timestamp': time.time(),
            'source_text': source_text,
            'summaries': [
                {
                    'summary_id': f"{task_id}_summary_{i}",
                    'model': summary.get('model', 'unknown'),
                    'summary_text': summary.get('summary', ''),
                    'evaluation_scores': {metric: None for metric in self.evaluation_schema},
                    'comments': ""
                }
                for i, summary in enumerate(summaries)
            ]
        }
        task_file = self.output_dir / f"evaluation_task_{task_id}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task, f, ensure_ascii=False, indent=2)
        logger.info(f"Tâche d’évaluation humaine créée: {task_file}")
        return task

    def load_evaluation_results(self, task_id: str) -> Optional[Dict]:
        """
        Charge le fichier JSON d’annotations humaines.
        """
        task_file = self.output_dir / f"evaluation_task_{task_id}.json"
        if not task_file.exists():
            logger.error(f"Tâche non trouvée: {task_id}")
            return None
        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_human_evaluations(self, task_ids: List[str]) -> Dict:
        """
        Fait des stats simples sur les évaluations humaines.
        """
        all_scores = []
        model_scores = {}

        for task_id in task_ids:
            task = self.load_evaluation_results(task_id)
            if not task:
                continue
            for summary in task['summaries']:
                model = summary['model']
                scores = summary['evaluation_scores']
                valid_scores = {k: v for k, v in scores.items() if v is not None}
                if valid_scores:
                    all_scores.append(valid_scores)
                    model_scores.setdefault(model, []).append(valid_scores)

        # Stats globales
        results = {
            'total_evaluations': len(all_scores),
            'metrics_stats': {},
            'model_comparison': {}
        }
        for metric in self.evaluation_schema:
            metric_values = [s[metric] for s in all_scores if metric in s]
            if metric_values:
                results['metrics_stats'][metric] = {
                    'mean': float(np.mean(metric_values)),
                    'std': float(np.std(metric_values)),
                    'median': float(np.median(metric_values)),
                    'min': float(np.min(metric_values)),
                    'max': float(np.max(metric_values)),
                    'count': len(metric_values)
                }

        for model, scores_list in model_scores.items():
            model_stats = {}
            for metric in self.evaluation_schema:
                metric_values = [s[metric] for s in scores_list if metric in s]
                if metric_values:
                    model_stats[metric] = {
                        'mean': float(np.mean(metric_values)),
                        'std': float(np.std(metric_values)),
                        'count': len(metric_values)
                    }
            results['model_comparison'][model] = model_stats

        return results

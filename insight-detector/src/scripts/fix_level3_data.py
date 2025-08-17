# fix_level3_data.py
"""
Script pour corriger les données du notebook Level 3
Récupère les vrais summaries depuis batch_summary_production.csv
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json

# Ajouter les chemins pour imports depuis src/scripts/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def fix_level3_data():
    """Corrige les données placeholder du Level 3"""
    
    print("CORRECTION DES DONNEES LEVEL 3")
    print("=" * 50)
    
    # 1. Chargement des cas CRITICAL
    critical_path = PROJECT_ROOT / 'data/detection/level2_simplified_priority_cases_for_level3.csv'
    critical_df = pd.read_csv(critical_path)
    print(f"Cas CRITICAL charges: {len(critical_df)}")
    
    # 2. Chargement des vrais summaries
    summaries_path = PROJECT_ROOT / 'data/results/batch_summary_production.csv'
    summaries_df = pd.read_csv(summaries_path)
    print(f"Summaries reels charges: {len(summaries_df)}")
    
    # 3. Mapping des IDs vers les vrais textes
    summary_mapping = {}
    for idx, row in summaries_df.iterrows():
        # ID format: text_id + strategy = summary_id
        summary_id = f"{row['text_id']}_confidence_weighted"
        summary_mapping[summary_id] = {
            'summary': row['summary'],
            'length': len(row['summary'].split()),
            'original_factuality': row['factuality'],
            'original_coherence': row['coherence']
        }
    
    # 4. Création des cas réels pour Level 3
    real_critical_cases = []
    found_count = 0
    
    for idx, row in critical_df.iterrows():
        summary_id = row['summary_id']
        
        if summary_id in summary_mapping:
            real_data = summary_mapping[summary_id]
            case = {
                'summary_id': summary_id,
                'summary': real_data['summary'],  # ✅ VRAI TEXTE
                'coherence_score': row['coherence_score'],
                'factuality_score': row['factuality_score'], 
                'validation_confidence': row['validation_confidence'],
                'issues_count': row['issues_count'],
                'justification': row['justification'],
                'original_length': real_data['length']
            }
            real_critical_cases.append(case)
            found_count += 1
        else:
            print(f"WARNING: Summary non trouve: {summary_id}")
    
    print(f"SUCCESS: {found_count}/{len(critical_df)} summaries reels recuperes")
    
    # 5. Sauvegarde des données corrigées
    output_path = 'data/detection/level3_real_critical_cases.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(real_critical_cases, f, indent=2, ensure_ascii=False)
    
    print(f"SAUVE: Donnees corrigees sauvees: {output_path}")
    
    # 6. Échantillon pour vérification
    print(f"\nECHANTILLON CORRIGE:")
    print("-" * 30)
    
    sample = real_critical_cases[0]
    print(f"ID: {sample['summary_id']}")
    print(f"Coherence: {sample['coherence_score']:.3f}")
    print(f"Factuality: {sample['factuality_score']:.3f}")
    print(f"Longueur: {sample['original_length']} mots")
    print(f"Summary (50 premiers chars): {sample['summary'][:50]}...")
    
    return real_critical_cases

if __name__ == "__main__":
    real_cases = fix_level3_data()
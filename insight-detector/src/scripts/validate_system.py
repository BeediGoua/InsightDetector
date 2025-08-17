#!/usr/bin/env python3
"""
Script de validation complète du système InsightDetector
Vérifie tous les composants critiques avant le passage à la Phase 4
"""

import sys
import json
import importlib
from pathlib import Path
import pandas as pd
from datetime import datetime

# Configuration des chemins pour exécution depuis src/scripts/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def print_status(message, status="INFO"):
    """Affichage formaté des statuts"""
    symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_dependencies():
    """Vérification des dépendances critiques"""
    print_status("=== VÉRIFICATION DES DÉPENDANCES ===")
    
    required_packages = [
        'torch', 'transformers', 'sentence_transformers',
        'rouge_score', 'bert_score', 'nltk', 'spacy',
        'streamlit', 'plotly', 'ipywidgets'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print_status(f"{package} installé", "SUCCESS")
        except ImportError:
            print_status(f"{package} MANQUANT", "ERROR")
            missing.append(package)
    
    if missing:
        print_status(f"Installer: pip install {' '.join(missing)}", "WARNING")
        return False
    return True

def check_models():
    """Vérification des modèles ML"""
    print_status("=== VÉRIFICATION DES MODÈLES ===")
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from models.abstractive_models import AbstractiveEnsemble
        
        ensemble = AbstractiveEnsemble(device="cpu")
        load_results = ensemble.load_models(["barthez", "french_t5"], max_models=2)
        
        loaded_count = sum(load_results.values())
        total_count = len(load_results)
        
        print_status(f"Modèles chargés: {loaded_count}/{total_count}", 
                    "SUCCESS" if loaded_count > 1 else "WARNING")
        
        for model, success in load_results.items():
            status = "SUCCESS" if success else "ERROR"
            print_status(f"  - {model}", status)
            
        return loaded_count > 0
        
    except Exception as e:
        print_status(f"Erreur modèles: {e}", "ERROR")
        return False

def check_data():
    """Vérification des données"""
    print_status("=== VÉRIFICATION DES DONNÉES ===")
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / "data" / "results"
    
    # Fichier principal
    data_file = results_dir / "all_summaries_and_scores.json"
    if not data_file.exists():
        print_status(f"Fichier principal manquant: {data_file}", "ERROR")
        return False
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summaries = data.get("summaries", [])
        evaluations = data.get("evaluations", [])
        
        print_status(f"Résumés: {len(summaries)}", "SUCCESS" if summaries else "ERROR")
        print_status(f"Évaluations: {len(evaluations)}", "SUCCESS" if evaluations else "ERROR")
        
        # Vérification métriques ROUGE/BERTScore
        if evaluations:
            df = pd.DataFrame(evaluations)
            rouge_valid = df['ROUGE-1 (f)'].notna().sum()
            bert_valid = df['BERTScore (f1)'].notna().sum()
            
            print_status(f"ROUGE valides: {rouge_valid}/{len(evaluations)}", 
                        "SUCCESS" if rouge_valid > len(evaluations)//2 else "WARNING")
            print_status(f"BERTScore valides: {bert_valid}/{len(evaluations)}", 
                        "SUCCESS" if bert_valid > len(evaluations)//2 else "WARNING")
            
            # Statistiques de cohérence
            coherence_mean = df['Cohérence'].mean()
            coherence_low = (df['Cohérence'] < 0.3).sum()
            
            print_status(f"Cohérence moyenne: {coherence_mean:.3f}", 
                        "SUCCESS" if coherence_mean > 0.4 else "WARNING")
            print_status(f"Résumés cohérence < 0.3: {coherence_low}", 
                        "WARNING" if coherence_low > len(evaluations)//4 else "SUCCESS")
        
        return len(summaries) > 0 and len(evaluations) > 0
        
    except Exception as e:
        print_status(f"Erreur lecture données: {e}", "ERROR")
        return False

def check_interfaces():
    """Vérification des interfaces"""
    print_status("=== VÉRIFICATION DES INTERFACES ===")
    
    base_dir = Path(__file__).parent
    
    # Dashboard Streamlit
    dashboard_file = base_dir / "validation_dashboard.py"
    if dashboard_file.exists():
        print_status("Dashboard Streamlit présent", "SUCCESS")
    else:
        print_status("Dashboard Streamlit manquant", "ERROR")
    
    # Notebook d'annotation
    annotation_file = base_dir / "notebooks" / "04_orchestration" / "annotation_tool.ipynb"
    if annotation_file.exists():
        print_status("Outil d'annotation présent", "SUCCESS")
    else:
        print_status("Outil d'annotation manquant", "ERROR")
    
    # Annotations humaines
    annotations_dir = base_dir / "data" / "results" / "human_annotations"
    if annotations_dir.exists():
        annotation_files = list(annotations_dir.glob("annotations_*.json"))
        if annotation_files:
            with open(max(annotation_files), 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            print_status(f"Annotations humaines: {len(annotations)}", 
                        "SUCCESS" if len(annotations) >= 10 else "WARNING")
        else:
            print_status("Aucune annotation humaine", "WARNING")
    else:
        print_status("Répertoire annotations manquant", "WARNING")
    
    return True

def generate_report():
    """Génération du rapport de validation"""
    print_status("=== RAPPORT DE VALIDATION ===")
    
    checks = {
        "Dépendances": check_dependencies(),
        "Modèles ML": check_models(), 
        "Données": check_data(),
        "Interfaces": check_interfaces()
    }
    
    passed = sum(checks.values())
    total = len(checks)
    
    print_status(f"\nRÉSULTAT GLOBAL: {passed}/{total} vérifications réussies")
    
    if passed == total:
        print_status("🎉 SYSTÈME PRÊT POUR PHASE 4", "SUCCESS")
        next_steps = [
            "1. Lancer le dashboard: streamlit run validation_dashboard.py",
            "2. Compléter les annotations humaines (20+ recommandé)",
            "3. Corriger la cohérence si < 0.4",
            "4. Implémenter la détection d'hallucinations"
        ]
    elif passed >= total * 0.75:
        print_status("⚠️ SYSTÈME PARTIELLEMENT PRÊT", "WARNING")
        next_steps = [
            "1. Corriger les erreurs critiques",
            "2. Relancer le notebook d'orchestration si nécessaire",
            "3. Vérifier l'installation des dépendances manquantes"
        ]
    else:
        print_status("❌ SYSTÈME NON PRÊT", "ERROR")
        next_steps = [
            "1. Résoudre tous les problèmes critiques",
            "2. Réinstaller les dépendances: pip install -r requirements.txt",
            "3. Relancer complètement le notebook d'orchestration"
        ]
    
    print_status("\n🔄 PROCHAINES ÉTAPES RECOMMANDÉES:")
    for step in next_steps:
        print_status(f"   {step}")
    
    # Sauvegarde du rapport
    report = {
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "score": passed / total,
        "next_steps": next_steps
    }
    
    report_file = Path(__file__).parent / "data" / "results" / "validation_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print_status(f"Rapport sauvegardé: {report_file}")
    
    return passed / total

if __name__ == "__main__":
    print("🧠 InsightDetector - Validation Système")
    print("=" * 50)
    
    score = generate_report()
    
    # Code de sortie pour scripts automatisés
    sys.exit(0 if score >= 0.75 else 1)
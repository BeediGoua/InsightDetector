#!/usr/bin/env python3
"""
Script de nettoyage intelligent pour redémarrage pipeline enhanced.

Supprime les données intermédiaires des niveaux 1-3 tout en conservant
les sources et données de base nécessaires au redémarrage.
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_backup(source_path: Path, backup_dir: Path):
    """Crée une sauvegarde avant suppression"""
    if source_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.name}_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        if source_path.is_file():
            shutil.copy2(source_path, backup_path)
        else:
            shutil.copytree(source_path, backup_path)
        
        print(f"✅ Backup créé: {backup_path}")
        return backup_path
    return None

def safe_remove(path: Path, description: str):
    """Suppression sécurisée avec confirmation"""
    if path.exists():
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            print(f"🗑️  Supprimé: {description}")
            return True
        except Exception as e:
            print(f"❌ Erreur suppression {description}: {e}")
            return False
    else:
        print(f"⚠️  Déjà absent: {description}")
        return True

def clean_for_enhanced_restart():
    """Nettoyage intelligent pour redémarrage pipeline enhanced"""
    
    print("🧹 NETTOYAGE INTELLIGENT POUR PIPELINE ENHANCED")
    print("=" * 60)
    
    # Créer dossier backup
    backup_dir = PROJECT_ROOT / "data" / "backups" / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Dossier backup: {backup_dir}")
    print()
    
    # === ÉTAPE 1: BACKUP DES DONNÉES IMPORTANTES ===
    print("📦 ÉTAPE 1: Backup des données importantes...")
    
    important_files = [
        PROJECT_ROOT / "data" / "results" / "batch_summary_production.csv",
        PROJECT_ROOT / "data" / "results" / "final_summary_production.json",
        PROJECT_ROOT / "data" / "exports" / "raw_articles.json",
        PROJECT_ROOT / "outputs" / "all_summaries_production.json"
    ]
    
    for file_path in important_files:
        if file_path.exists():
            create_backup(file_path, backup_dir)
    
    print()
    
    # === ÉTAPE 2: SUPPRESSION NIVEAU 1 (ANCIEN) ===
    print("🗑️  ÉTAPE 2: Suppression données Niveau 1 (ancien système)...")
    
    level1_patterns = [
        "level1_*.csv",
        "level1_*.json"
    ]
    
    detection_dir = PROJECT_ROOT / "data" / "detection"
    for pattern in level1_patterns:
        for file_path in detection_dir.glob(pattern):
            safe_remove(file_path, f"Niveau 1: {file_path.name}")
    
    print()
    
    # === ÉTAPE 3: SUPPRESSION NIVEAU 2 (ANCIEN) ===
    print("🗑️  ÉTAPE 3: Suppression données Niveau 2 (ancien système)...")
    
    level2_files = [
        detection_dir / "level2_simplified_priority_cases_with_ids.csv",
        detection_dir / "level2_simplified_results_with_ids.csv",
        PROJECT_ROOT / "outputs" / "level2_output_with_source_id.json",
        PROJECT_ROOT / "outputs" / "level2_simplified_priority_cases_with_ids.csv",
        PROJECT_ROOT / "outputs" / "level2_simplified_results_with_ids.csv"
    ]
    
    for file_path in level2_files:
        safe_remove(file_path, f"Niveau 2: {file_path.name}")
    
    print()
    
    # === ÉTAPE 4: SUPPRESSION NIVEAU 3 (ANCIEN) ===
    print("🗑️  ÉTAPE 4: Suppression données Niveau 3 (ancien système)...")
    
    level3_dirs = [
        PROJECT_ROOT / "data" / "processed" / "level3",
        PROJECT_ROOT / "outputs" / "level3"
    ]
    
    for dir_path in level3_dirs:
        safe_remove(dir_path, f"Niveau 3: {dir_path}")
    
    print()
    
    # === ÉTAPE 5: SUPPRESSION MAPPINGS ANCIENS ===
    print("🗑️  ÉTAPE 5: Suppression mappings anciens...")
    
    mapping_files = list((PROJECT_ROOT / "outputs").glob("mapping_*.csv"))
    for file_path in mapping_files:
        safe_remove(file_path, f"Mapping: {file_path.name}")
    
    print()
    
    # === ÉTAPE 6: VÉRIFICATION CONSERVATION ===
    print("✅ ÉTAPE 6: Vérification conservation données sources...")
    
    essential_files = [
        "data/exports/raw_articles.json",
        "data/exports/enriched_articles.json", 
        "data/results/batch_summary_production.csv",
        "data/results/final_summary_production.json"
    ]
    
    all_preserved = True
    for rel_path in essential_files:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            print(f"✅ Conservé: {rel_path}")
        else:
            print(f"❌ MANQUANT: {rel_path}")
            all_preserved = False
    
    print()
    
    # === RÉSUMÉ ===
    print("📊 RÉSUMÉ DU NETTOYAGE")
    print("-" * 40)
    
    if all_preserved:
        print("✅ Toutes les données sources sont conservées")
        print("✅ Données intermédiaires supprimées")
        print("✅ Pipeline prêt pour redémarrage enhanced")
        print()
        print("🚀 PROCHAINES ÉTAPES:")
        print("1. Exécuter: notebooks/05_detection_hallucinations/01_niveau0_prefilter_test.ipynb")
        print("2. Puis: notebooks/05_detection_hallucinations/02_niveau1_heuristic_test.ipynb")
        print("3. Puis: notebooks/05_detection_hallucinations/03_niveau2.ipynb")
        print("4. Enfin: notebooks/05_detection_hallucinations/04_niveau3_orchestrator.ipynb")
        print()
        print("💡 Les composants ENHANCED seront utilisés automatiquement!")
    else:
        print("❌ ATTENTION: Certaines données sources manquent")
        print("⚠️  Vérifiez le backup avant de continuer")
    
    print(f"📦 Backup disponible dans: {backup_dir}")

if __name__ == "__main__":
    # Demander confirmation
    print("🚨 ATTENTION: Ce script va supprimer les données intermédiaires des niveaux 1-3")
    print("   Les données sources seront conservées et un backup sera créé.")
    print()
    
    response = input("Continuer? (oui/non): ").lower().strip()
    if response in ['oui', 'o', 'yes', 'y']:
        clean_for_enhanced_restart()
    else:
        print("❌ Nettoyage annulé")
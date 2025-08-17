  📋 ORDRE D'EXÉCUTION ET VÉRIFICATIONS

  Voici l'ordre complet d'exécution des fichiers avec      
  les vérifications requises avant chaque étape :

  🔧 Phase de Vérification Préalable

  Avant tout lancement, vérifiez que les composants        
  enhanced sont disponibles :

  # Vérifications des fichiers enhanced
  ls src/detection/level0_prefilter_enhanced.py    #       
  ✅ Doit exister
  ls src/detection/level1_heuristic.py             #       
  ✅ Mis à jour avec version enhanced
  ls src/detection/level2_intelligent/             #       
  ✅ Dossier avec version enhanced
  ls src/detection/level3_adaptive/                #       
  ✅ Dossier avec version enhanced

  ---
  📊 1. NIVEAU 0 - Préfiltre Enhanced

  Fichier : notebooks/05_detection_hallucinations/01_n     
  iveau0_prefilter_test.ipynb

  Vérifications avant lancement :
  - ✅ data/results/batch_summary_production.csv
  existe (372 résumés)
  - ✅ src/detection/level0_prefilter_enhanced.py est      
  présent
  - ✅ Répertoire data/detection/ existe pour
  sauvegardes

  Commande :
  # Exécuter le notebook niveau 0
  jupyter notebook notebooks/05_detection_hallucinatio     
  ns/01_niveau0_prefilter_test.ipynb

  Attendu en sortie :
  - Fichier data/detection/level0_filter_results.csv       
  - Fichier data/detection/level0_filter_stats.json        
  - Affichage : "NIVEAU 0 ENHANCED importé" (si
  composants enhanced disponibles)

  ---
  🔍 2. NIVEAU 1 - Heuristique Enhanced

  Fichier : notebooks/05_detection_hallucinations/02_n     
  iveau1_heuristic_test.ipynb

  Vérifications avant lancement :
  - ✅ Résultats niveau 0 disponibles dans
  data/detection/
  - ✅ src/detection/level1_heuristic.py mis à jour        
  avec version enhanced
  - ✅ SpaCy installé (optionnel) : python -c "import      
  spacy"

  Commande :
  # Exécuter le notebook niveau 1
  jupyter notebook notebooks/05_detection_hallucinatio     
  ns/02_niveau1_heuristic_test.ipynb

  Attendu en sortie :
  - Affichage : "NIVEAU 1 ENHANCED importé (analyseur      
  amélioré avec aliases)"
  - Détection des patterns corruption
  confidence_weighted
  - Seuils longueur corrigés (15-200 mots vs 400-500       
  original)

  ---
  🧠 3. NIVEAU 2 - Classification Intelligente

  Fichier : notebooks/05_detection_hallucinations/03_n     
  iveau2.ipynb

  Vérifications avant lancement :
  - ✅ Résultats niveau 1 dans
  data/detection/level1_usable.csv
  - ✅ Articles sources dans
  data/exports/raw_articles.json
  - ✅ Mapping files dans outputs/mapping_*.csv
  - ✅ src/detection/level2_intelligent/ existe

  Commande :
  # Exécuter le notebook niveau 2
  jupyter notebook notebooks/05_detection_hallucinatio     
  ns/03_niveau2.ipynb

  Attendu en sortie :
  - Affichage : "NIVEAU 2 ENHANCED importé
  (classification intelligente)"
  - Classification CRITICAL différenciée
  (RECOVERABLE/HALLUCINATION/CORRUPTED)
  - Fichiers dans outputs/level2_*.csv et
  outputs/level2_output_with_source_id.json

  ---
  🚀 4. NIVEAU 3 - Stratégies Adaptatives

  Fichier : notebooks/05_detection_hallucinations/04_n     
  iveau3_orchestrator.ipynb

  Vérifications avant lancement :
  - ✅ Résultats niveau 2 dans
  outputs/level2_simplified_results_with_ids.csv
  - ✅ outputs/level2_output_with_source_id.json
  existe
  - ✅ Mapping files dans outputs/mapping_*.csv
  - ✅ src/detection/level3_adaptive/ existe
  - ✅ Configuration YAML dans
  src/detection/level3_adaptive/config/level3.yaml

  Commande :
  # Exécuter le notebook niveau 3
  jupyter notebook notebooks/05_detection_hallucinatio     
  ns/04_niveau3_orchestrator.ipynb

  Attendu en sortie :
  - Affichage : "NIVEAU 3 ENHANCED importé (stratégies     
   adaptatives)"
  - Stratégies adaptatives : edit_intelligent,
  regenerate_from_source, escalate_manual
  - Taux d'acceptation >0% (vs 0% original)
  - Fichiers dans
  outputs/level3/exports/level3_results.csv

  ---
  🎯 Ordre de Lancement Complet

  Si tu relances tout de zéro (recommandé après 
  migration) :

  # 1. Nettoyer les anciens résultats (optionnel)
  rm -rf data/detection/* outputs/*

  # 2. Lancer dans l'ordre avec vérifications
  echo "🔧 Lancement Niveau 0..."
  jupyter nbconvert --execute notebooks/05_detection_h     
  allucinations/01_niveau0_prefilter_test.ipynb

  echo "🔍 Lancement Niveau 1..."
  jupyter nbconvert --execute notebooks/05_detection_h     
  allucinations/02_niveau1_heuristic_test.ipynb

  echo "🧠 Lancement Niveau 2..."
  jupyter nbconvert --execute notebooks/05_detection_h     
  allucinations/03_niveau2.ipynb

  echo "🚀 Lancement Niveau 3..."
  jupyter nbconvert --execute notebooks/05_detection_h     
  allucinations/04_niveau3_orchestrator.ipynb

  echo "✅ Pipeline ENHANCED complet terminé!"

  ---
  📈 Métriques de Succès Attendues

  Niveau 0 : 85-90% de résumés validés (vs seuils plus     
   stricts enhanced)Niveau 1 : Détection patterns
  confidence_weighted, seuils réalistesNiveau 2 :
  Classification intelligente avec sous-types
  CRITICALNiveau 3 : ≥70% acceptation (vs 0% système       
  original) 🎉

  ---
  ⚠️ Points d'Attention

  1. Imports Enhanced : Chaque notebook devrait
  afficher "ENHANCED importé" au lieu d'alertes de
  fallback
  2. Données Manquantes : Si erreurs de fichiers
  manquants, revenir aux étapes précédentes
  3. Compatibilité : Les notebooks gardent le même
  fonctionnement mais avec composants améliorés
  4. Performance : Le niveau 3 enhanced devrait
  résoudre le blocage 0% → taux élevé

  La migration intelligente est terminée ! Tu peux         
  maintenant exécuter le pipeline complet avec les         
  composants enhanced qui résolvent les problèmes 
  identifiés.
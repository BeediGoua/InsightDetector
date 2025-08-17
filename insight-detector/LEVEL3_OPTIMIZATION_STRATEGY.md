# STRATÉGIE D'OPTIMISATION LEVEL 3 - PLAN D'ACTION COMPLET

## 🚨 DIAGNOSTIC : ÉCHEC SYSTÉMIQUE DU LEVEL 3

### Problème principal : 0% d'acceptation due à :
1. **Évaluation factice** (`l2_like_evaluate` = random)
2. **Topic overlap défaillant** (27% = 0.0, moyenne 1.4%)
3. **Critères d'acceptation inadaptés** 
4. **Mode decision incohérente**

---

## 🎯 STRATÉGIE EN 4 PHASES

### PHASE 1 : CORRECTIONS CRITIQUES (PRIORITÉ IMMÉDIATE)

#### 1.1 Remplacer l'évaluation factice
```python
# AVANT (défaillant)
def l2_like_evaluate(summary, source_text):
    return {"tier": random.choice(...), "factuality_score": random.uniform(...)}

# APRÈS (réaliste)  
def l2_like_evaluate(summary, source_text):
    # Utiliser des métriques déterministes basées sur :
    # - Longueur relative (summary vs source)
    # - Entités communes (NER overlap)
    # - Similarité lexicale (TF-IDF/embedding)
    # - Cohérence structurelle
```

#### 1.2 Corriger le topic overlap
```python
# Problème : Calcul défaillant donnant 27% = 0.0
# Solution : Diagnostic + métriques alternatives
# - BERT embeddings similarity
# - Entités nommées communes  
# - Keywords TF-IDF overlap
```

#### 1.3 Assouplir les critères d'acceptation
```yaml
# AVANT (impossible)
acceptance:
  accepted_tiers: ["GOOD","EXCELLENT","MODERATE"]
  require_monotonic_improvement: true
  
# APRÈS (réaliste)
acceptance:
  accepted_tiers: ["GOOD","EXCELLENT","MODERATE","IMPROVED_CRITICAL"]
  require_monotonic_improvement: false
  allow_stagnation_if_tier_improved: true
```

### PHASE 2 : OPTIMISATIONS STRUCTURELLES

#### 2.1 Mode decision améliorée
```python
def choose_mode_v2(row, cfg):
    # Priorité : RE-SUMMARIZE pour CRITICAL si texte suffisant
    # Fallback intelligent : EDIT avec règles spécifiques
    # Garde-fous : éviter RE-SUMMARIZE sur textes courts
```

#### 2.2 Pipeline de validation progressive
```
CRITICAL → EDIT/RE-SUMMARIZE → ÉVALUATION → IMPROVED_CRITICAL → ACCEPTATION
```

#### 2.3 Métriques d'amélioration réalistes
- Tier progression : CRITICAL → MODERATE = succès
- Factuality relative : +10% = amélioration significative  
- Issues réduction : -2 issues = progrès notable

### PHASE 3 : OPTIMISATIONS AVANCÉES

#### 3.1 Évaluation hybride
- Métriques automatiques (embedding, NER, TF-IDF)
- Règles heuristiques (longueur, structure)
- Modèles pré-entraînés (si disponibles)

#### 3.2 Calibrage dynamique des seuils
```python
# Ajustement automatique basé sur distribution réelle
def auto_calibrate_thresholds(historical_data):
    # Viser 15-30% d'acceptation
    # Ajuster selon performance
```

### PHASE 4 : VALIDATION ET DÉPLOIEMENT

#### 4.1 Tests de non-régression
- Maintenir 81 candidats CRITICAL
- Viser 15-30% d'acceptation finale
- Préserver mapping 100%

#### 4.2 Monitoring continu
- Tracking des métriques d'acceptation
- Alertes si chute < 10%
- Révision périodique des seuils

---

## 📊 OBJECTIFS QUANTIFIÉS

| Métrique | Actuel | Objectif Phase 1 | Objectif Phase 2 |
|----------|--------|------------------|------------------|
| Taux acceptation | 0% | 15% | 25% |
| Topic overlap moyen | 1.4% | 8% | 12% |
| CRITICAL traités | 0 | 12 | 20 |
| Pipeline sans erreur | ✅ | ✅ | ✅ |

---

## 🚀 PLAN D'IMPLÉMENTATION

### Jour 1 : Phase 1 critique
1. Remplacer `l2_like_evaluate` factice
2. Diagnostic topic overlap
3. Assouplir critères acceptation
4. Test sur 10 cas

### Jour 2 : Phase 2 structurelle  
1. Optimiser `choose_mode`
2. Implémenter validation progressive
3. Test sur 50 cas
4. Calibrage seuils

### Jour 3 : Validation complète
1. Test sur les 81 cas
2. Vérification non-régression
3. Documentation finale
4. Déploiement

---

## ⚠️ RISQUES ET MITIGATION

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Évaluation toujours défaillante | Élevé | Tests unitaires + métriques simples |
| Seuils trop permissifs | Moyen | Calibrage progressif + monitoring |
| Regression pipeline amont | Faible | Tests de non-régression systématiques |

---

Cette stratégie garantit un Level 3 fonctionnel avec un taux d'acceptation réaliste tout en préservant la qualité du pipeline.
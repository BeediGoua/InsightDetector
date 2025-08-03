

# PHASE 4 : Détection d’hallucinations – Architecture Complète

## Contexte et Objectif

Le pipeline actuel génère **372 résumés évalués** avec des métriques de qualité.
L’analyse met en évidence :

* **64 résumés de grade D** (≈17%) confirmés comme hallucinations.
* **Patterns fréquents** : cohérence <0.3, présence de métadonnées parasites, répétitions, omissions critiques.

**Objectif global :**
Mettre en place un système de détection d’hallucinations **modulaire à 4 niveaux**, visant :

* **Précision >90%**
* **Rappel >85%**
* **Latence <2s** par résumé

---

## Architecture Détection 4-Niveaux

**🚀 Innovation clé :** Système de calibrage automatique adaptatif qui analyse les données réelles pour optimiser les seuils de détection à chaque niveau, garantissant une précision maximale tout en minimisant les faux positifs.

**Principe :** Chaque niveau peut s'auto-calibrer en analysant :
- Les **distributions statistiques** des données d'entraînement
- Les **grades de qualité** existants (A+, A, B+, B, C, D)
- Les **patterns d'erreurs** identifiés dans les cas problématiques
- Les **métriques de performance** (temps de traitement, taux de rejet)

### Niveau 0 – Pré-filtrage qualité (<50ms)

**But :** Éliminer les cas manifestement problématiques avant toute analyse coûteuse.

**🎯 Innovation : Calibrage automatique des seuils**
Le système analyse automatiquement les données d'entraînement pour déterminer les seuils optimaux :
* **3 stratégies** : percentiles conservateurs, analyse grades qualité, détection outliers (IQR)
* **Sélection intelligente** : priorise l'analyse des grades D (problématiques)
* **Validation cohérence** : vérifie la logique des seuils avant application
* **Estimation impact** : calcule le taux de rejet attendu

**Critères d'exclusion (auto-calibrés sur 372 résumés) :**

* **Longueur anormale** : Seuils adaptatifs basés sur percentiles et grades D
  - Mode manuel : <10 mots ou >600 mots (estimation conservative)
  - Mode auto-calibré : seuils optimisés selon distribution réelle des données
* **Répétitions excessives** : Algorithme intelligent avec seuils adaptatifs (phrases >4x, séquences >3x selon contexte).
* **Métadonnées parasites** : "newsletter", "publicité", "s'abonner", "cookies", "GDPR" (17% des cas observés).
* **Anomalies d'encodage** ou caractères corrompus (détection via regex unicode).

**🔧 Usage :**
```python
# Calibrage automatique
filter_auto = auto_calibrate_filter(summaries_data_with_grades)
# OU seuils manuels
filter_manual = QualityFilter(min_words=10, max_words=600)
```

**Décision :**

* Cas invalides → rejet automatique
* Cas valides → passage au Niveau 1

**Performance :** <50ms par résumé, taux rejet optimal 5-8%

---

### Niveau 1 – Détection heuristique rapide (<100ms)

#### 1.1 Incohérences temporelles

* Détection d’anachronismes : *“Napoléon utilise un smartphone”*
* Chronologies impossibles : *“Après sa mort en 2020, il a déclaré en 2021”*
* Contradictions internes : “hier” vs “l’année dernière”

#### 1.2 Validation des entités

Triple vérification :

1. **NER local** (spaCy + CamemBERT-NER)
2. **Cross-check externe** (Wikidata, DBpedia)
3. **Proximité sémantique** via embeddings

Exemple :
“Le président Emmanuel Dupont” →

* Détection PERSON (NER)
* Vérification Wikidata ❌ (inexistant)
* Similarité avec Emmanuel Macron : 0.85
* Verdict : Hallucination probable

#### 1.3 Relations causales suspectes

* Vérification par graphe de connaissances (relations connues).
* Contrôle de plausibilité (*pluie → licenciements* = improbable).
* Détection de contradictions via NLI (CamemBERT fine-tuné).

---

### Niveau 2 – Validation factuelle approfondie (<1s)

#### 2.1 Fidélité sémantique multi-niveaux

* Similarité embeddings (Sentence-BERT).
* Alignement entités source ↔ résumé.
* Extraction et comparaison de triplets (sujet, prédicat, objet).
* Cohérence de ton et polarité.

**Score composite calculé :**

```python
fidélité_score = (
    0.3 * embedding_similarity +
    0.3 * entity_overlap +
    0.25 * fact_preservation +
    0.15 * sentiment_coherence
)
```

#### 2.2 Fact-checking multi-sources

* Wikidata (0.4) : entités, dates, relations
* DBpedia (0.3) : catégories et descriptions
* Google Fact Check (0.2) : actualités récentes
* Base métier interne (0.1) : contexte spécifique

#### 2.3 Détection par omission

* Vérifier que les entités et faits critiques ne sont pas supprimés.
* Identifier la perte de nuances (*“peut-être” → “certainement”*).
* Détecter les contextes omis (conditions, exceptions).

---

### Niveau 3 – Classificateur ML hybride (<2s)

#### Features (73 au total)

* **Linguistiques (25)** : perplexité, richesse lexicale, cohésion textuelle.
* **Factuelles (20)** : scores fact-checking, contradictions.
* **Sémantiques (15)** : similarité BERT multicouche, dérive de sujet.
* **Méta (13)** : longueur relative, stratégie de génération, confiance modèle.

#### Ensemble de modèles

* Random Forest (robustesse, interprétabilité).
* SVM RBF (détection de frontières complexes).
* CamemBERT fine-tuné (analyse contextuelle profonde).
* Réseau neuronal léger pour la fusion.

**Stratégie adaptative :**

* Si confiance élevée (RF + SVM), on prend le vote majoritaire.
* Sinon, fallback vers CamemBERT.

---

### Niveau 4 – Validation humaine ciblée

**Déclencheurs :**

* Conflits entre niveaux.
* Score final <0.8 en confiance.
* Domaines sensibles (finance, santé, politique).

**Interface prévue :**

* Passage suspect surligné.
* Contexte source affiché en parallèle.
* Boutons de décision rapide (valider / rejeter / incertain).
* Feedback intégré dans la boucle d’apprentissage actif.

---

## Plan d’Implémentation Progressif

* **Semaines 1–2** : Mise en place Niveau 0 + Niveau 1 (filtres rapides).
* **Semaines 3–4** : Ajout du Niveau 2 (validation factuelle de base).
* **Semaines 5–6** : Implémentation du Niveau 3 (classificateur ML).
* **Semaines 7–8** : Intégration Niveau 4 et tests complets en production.

---

## Métriques de Performance

**Techniques :**

* Précision détection >90%
* Rappel hallucinations >85%
* F1-score composite >87%
* Latence <2s

**Business :**

* Accord inter-annotateur κ >0.8
* Corrélation humain-modèle r >0.85
* Réduction du fact-checking manuel : 80%
* Coût par analyse <0.10 €

**Opérationnelles :**

* Traitement >1000 analyses/heure
* Disponibilité 99.5%
* Temps de réponse API p95 <3s
* Consommation mémoire <2GB par worker

---

## Innovations Clés

1. **Pipeline 4-niveaux** : équilibre entre rapidité et précision.
2. **Détection par omission** : une dimension rarement couverte.
3. **Validation humaine ciblée** : intervention minimale mais efficace.
4. **Features avancées** : couverture linguistique, factuelle et contextuelle.
5. **Ensemble hybride** : synergie entre heuristiques, ML classique et modèles de type transformers.

---

Cette architecture fournit une base robuste et évolutive pour détecter les hallucinations dans les résumés générés, avec un plan clair d’implémentation progressive et des métriques mesurables.



## L'idée générale

Ton projet s'appelle **InsightDetector**.
C'est un outil qui sert à **vérifier les textes générés par des IA** (comme ChatGPT ou BART), parce que ces IA écrivent parfois des phrases qui semblent vraies mais sont **fausses ou inventées**.
On appelle ça des **hallucinations**.

**But** : créer un système qui lit un texte, le résume et détecte automatiquement s'il contient des erreurs, incohérences ou inventions.

### Pourquoi ce projet ?

Dans le monde actuel, les IA génératives sont partout :
- **Journalisme** : Les journaux utilisent l'IA pour résumer des articles
- **Marketing** : Les entreprises génèrent du contenu automatiquement  
- **Recherche** : Les chercheurs s'appuient sur l'IA pour analyser des textes

**Le problème** : Ces IA peuvent inventer des faits, mélanger des dates, ou créer des liens causaux qui n'existent pas. Exemple typique : "Le président X a déclaré Y le 15 mars" alors que cette déclaration n'a jamais eu lieu.

**Ton solution** : Un système automatique qui agit comme un "fact-checker" intelligent, capable de dire "Attention, ce résumé contient probablement des erreurs".

---

## 🛠 Le pipeline du projet

Ton système marche comme une **chaîne d’étapes**, qu’on appelle un **pipeline**.
Voici les étapes :

```
Articles (sources RSS) 
   → Prétraitement 
   → Résumé automatique 
   → Évaluation & détection d’hallucinations 
   → Interface pour validation humaine
```

On va les détailler une par une.

---

### 1. Collecter des articles (Phase 1)

**Ce qu'on fait concrètement**
Tu as créé un script qui va chercher des **articles d'actualité** depuis des flux RSS (des listes d'articles fournies par les journaux).

**Étapes détaillées :**
1. **Configuration des sources RSS** : Tu as défini une liste de flux RSS fiables (Le Monde, Le Figaro, Reuters, etc.)
2. **Scraping automatique** : Le script `rss_collector.py` se connecte à chaque flux RSS
3. **Parsing XML/RSS** : Il analyse le format XML pour extraire titre, contenu, date, auteur
4. **Gestion d'erreurs** : Si un article ne se charge pas (erreur 404, timeout), le script continue sans s'arrêter
5. **Stockage JSON** : Chaque article est sauvé avec sa métadonnée dans `raw_articles.json`

**Résultats obtenus :**
* **547 articles** récoltés au total
* Sources diversifiées (actualités, tech, science, économie)
* Articles en français principalement
* Période de collecte : [dates que tu as utilisées]

**Difficultés rencontrées :**
- Certains sites bloquent les bots → solution : ajouter des headers HTTP réalistes
- Flux RSS parfois indisponibles → solution : retry automatique avec délais
- Formats RSS différents selon les sites → solution : parser flexible

**Pourquoi cette étape est cruciale ?**
Il faut un **corpus** (base de données de textes) varié et réaliste pour tester ton système. Sans données diversifiées, ton détecteur d'hallucinations ne sera pas robuste.

**Code principal utilisé :**
```python
# Dans rss_collector.py
import feedparser
import requests
import json

def collect_rss_articles(rss_urls):
    articles = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Extraction des métadonnées
                article = {
                    'title': entry.title,
                    'content': entry.summary,
                    'url': entry.link,
                    'published': entry.published,
                    'source': feed.feed.title
                }
                articles.append(article)
        except Exception as e:
            print(f"Erreur avec {url}: {e}")
    return articles
```

---

### 2.  Nettoyer et préparer les articles (Phase 2)

**Ce qu'on fait concrètement**
Cette phase est cruciale : on transforme des données "brutes" en données "exploitables" par les algorithmes.

**Étapes détaillées du preprocessing :**

1. **Détection de la langue**
   ```python
   from langdetect import detect
   
   def filter_french_articles(articles):
       french_articles = []
       for article in articles:
           try:
               if detect(article['content']) == 'fr':
                   french_articles.append(article)
           except:
               pass  # Ignore les textes trop courts pour détecter la langue
       return french_articles
   ```
   - **Pourquoi ?** Ton système est conçu pour le français. Garder des articles en anglais/espagnol causerait des erreurs dans l'analyse sémantique.

2. **Suppression des doublons intelligente**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics.pairwise import cosine_similarity
   
   def remove_duplicates(articles, threshold=0.8):
       # Calcule la similarité TF-IDF entre tous les articles
       vectorizer = TfidfVectorizer()
       tfidf_matrix = vectorizer.fit_transform([a['content'] for a in articles])
       similarity_matrix = cosine_similarity(tfidf_matrix)
       
       # Garde seulement les articles uniques
       unique_articles = []
       for i, article in enumerate(articles):
           is_duplicate = False
           for j in range(i):
               if similarity_matrix[i][j] > threshold:
                   is_duplicate = True
                   break
           if not is_duplicate:
               unique_articles.append(article)
       return unique_articles
   ```
   - **Pourquoi ?** Deux articles peuvent parler du même événement avec des mots différents. On utilise TF-IDF (fréquence des termes) pour détecter ces doublons sémantiques.

3. **Extraction d'entités nommées (NER)**
   ```python
   import spacy
   
   nlp = spacy.load("fr_core_news_sm")
   
   def extract_entities(text):
       doc = nlp(text)
       entities = {
           'PERSON': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
           'ORG': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
           'GPE': [ent.text for ent in doc.ents if ent.label_ == 'GPE'],  # Lieux
           'DATE': [ent.text for ent in doc.ents if ent.label_ == 'DATE']
       }
       return entities
   ```
   - **Pourquoi ?** Les entités sont cruciales pour détecter les hallucinations. Si un résumé change "Emmanuel Macron" en "François Mitterrand", c'est une erreur grave.

4. **Calcul de scores de qualité**
   ```python
   def calculate_quality_scores(article):
       content = article['content']
       
       # Score de lisibilité (indices de Flesch)
       readability = textstat.flesch_reading_ease(content)
       
       # Score de richesse informationnelle
       entities = extract_entities(content)
       entity_density = sum(len(v) for v in entities.values()) / len(content.split())
       
       # Score de structure (présence intro/développement/conclusion)
       structure_score = analyze_text_structure(content)
       
       return {
           'readability': readability,
           'entity_density': entity_density,
           'structure': structure_score,
           'length': len(content.split()),
           'composite_score': (readability + entity_density*100 + structure_score) / 3
       }
   ```

**Résultats obtenus :**
* **Avant nettoyage** : 547 articles bruts
* **Après filtrage langue** : 398 articles français
* **Après suppression doublons** : 186 articles uniques  
* **Fichier final** : `calibration_corpus_300.json` (enrichi avec métadonnées)

**Difficultés rencontrées :**
- **Détection de langue imparfaite** : Certains articles multilingues mal classés → solution : vérification manuelle des cas limites
- **Seuil de similarité délicat** : Trop bas = perte d'articles différents, trop haut = doublons restants → solution : tests A/B avec différents seuils
- **Entités mal reconnues** : SpaCy confond parfois prénoms/noms de lieux → solution : post-processing manuel pour les entités critiques

**Pourquoi cette étape est essentielle ?**
Un système de résumé et de détection doit partir de **données propres et bien structurées**. C'est comme cuisiner : si tes ingrédients sont pourris, ton plat sera mauvais même avec la meilleure recette.

---

### 3. Générer des résumés automatiques (Phase 3)

**Ce qu’on fait**
Tu as développé un **moteur de résumé** (`summarizer_engine.py`).
Il essaie plusieurs modèles pour résumer un article :

1. **Résumé abstractive**

   * Le modèle **réécrit** le texte avec ses propres mots.
   * Exemple :
     Texte original →
     *“Microsoft a licencié 9 000 employés…”*
     Résumé →
     *“Microsoft a réduit ses effectifs de 9 000 personnes.”*

2. **Fallback extractif** (si le modèle 1 échoue)

   * On prend directement des phrases importantes du texte.

3. **Baseline LeadK** (si tout échoue)

   * On garde les 3 premières phrases.

Ensuite, tu **fusionnes les résumés** avec `ensemble_manager.py` :

* soit en donnant plus de poids au résumé le plus **confiant**,
* soit selon le **domaine** (ex. juridique, scientifique…),
* soit selon la **longueur du texte source**.

**Pourquoi ?**
Un modèle seul peut se tromper → tu combines plusieurs résumés pour être plus robuste.

---

### 4. Détecter les hallucinations (Phase 4 – prochaine étape)

C’est **le cœur de ton projet**.
Une fois que tu as un résumé, tu veux savoir : **est-il fiable ?**

Ton plan est d’avoir **3 niveaux de vérification** :

#### Niveau 1 – Vérification rapide

* Vérifie si le résumé reprend bien les mots du texte (ROUGE).
* Vérifie la similarité sémantique (BERTScore).
* Vérifie que les entités (noms, lieux, dates) n’ont pas changé.

#### Niveau 2 – Vérification factuelle

* Compare avec des bases de connaissances (Wikidata).
* Vérifie les chiffres et les dates.
* Détecte les contradictions logiques.

#### Niveau 3 – Analyse profonde

* Utilise un grand modèle (comme GPT-4) pour **juger la fiabilité**.
* Analyse plus subtile : contextes implicites, plausibilité globale.

**Types d’hallucinations détectées** :

* Mauvais nom de personne (**Entity\_Substitution**)
* Chiffres inventés (**Numerical\_Distortion**)
* Causes inventées (**Causal\_Invention**)
* Dates impossibles (**Temporal\_Inconsistency**)
* etc.

**Pourquoi ?**
Parce que la fluidité d’un texte ≠ vérité. Tu crées un **filet de sécurité**.

---

### 5. Interface utilisateur (Phase 5)

**Ce qu’on fait**
Tu as commencé un **dashboard Streamlit** :

* Pour afficher les articles et résumés.
* Pour montrer les scores (ROUGE, BERTScore, etc.).
* Pour permettre aux humains de valider ou corriger.

Exemple :
Un journaliste peut charger un article → voir le résumé + score de factualité → décider si le publier ou non.

**Pourquoi ?**
Les utilisateurs finaux (journalistes, analystes, entreprises) doivent **voir et comprendre les résultats**.

---

### 6. Déploiement cloud (Phase 6)

**Ce qui est prévu**

* Rendre InsightDetector disponible via une API (FastAPI).
* Hébergement sur AWS/GCP.
* Monitoring (temps de réponse, erreurs).
* Sécurité (authentification, quotas).

**Pourquoi ?**
Passer d’un prototype de recherche → à un **outil utilisable en entreprise**.

---

##  Métriques et évaluation

Tu ne fais pas que générer des résumés, tu les **notes** avec des métriques :

* **ROUGE** : recouvrement lexical.
* **BERTScore** : similarité sémantique.
* **Factualité** : exactitude des faits.
* **Cohérence** : logique du texte.
* **Lisibilité** : fluidité de lecture.
* **Score composite** : mélange des critères.

Tu visualises ces scores avec des **graphiques (barplots, heatmaps)** pour comparer plusieurs résumés.

---

## Concrètement, tu fais donc :

1. **Tu récoltes des articles**
   → comme construire ta bibliothèque.

2. **Tu les nettoies et tu choisis les meilleurs**
   → tu enlèves les doublons et tu gardes les textes fiables.

3. **Tu les résumes automatiquement**
   → tu essaies plusieurs modèles, avec des plans B si ça échoue.

4. **Tu prépares la détection d’hallucinations**
   → en construisant un système multi-niveaux pour repérer les erreurs.

5. **Tu crées une interface pour les humains**
   → afin qu’ils puissent vérifier et corriger facilement.

6. **Tu planifies le déploiement cloud**
   → pour qu’on puisse utiliser ton système partout et rapidement.

 
"""
Implémentation modèles extractifs : CamemBERT, TextRank, TF-IDF+MMR
avec algorithmes de sélection optimisés et diversité garantie
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
import re
import spacy
from typing import List, Dict, Tuple, Optional
import logging
import time
from collections import Counter

# Téléchargement NLTK si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractiveModelBase:
    """Classe de base pour modèles extractifs"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False
    
    def load_model(self) -> bool:
        """Chargement du modèle"""
        raise NotImplementedError
    
    def extract_sentences(self, text: str, num_sentences: int = 3) -> Dict:
        """Extraction des meilleures phrases"""
        raise NotImplementedError
    
    def _split_sentences(self, text: str) -> List[str]:
        """Segmentation robuste en phrases"""
        # Méthode hybride : NLTK + regex pour robustesse
        try:
            sentences = nltk.sent_tokenize(text, language='french')
        except:
            # Fallback regex si NLTK échoue
            sentences = re.split(r'[.!?]+', text)
        
        # Nettoyage et filtrage
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filtrer phrases trop courtes/longues
            if 20 <= len(sentence) <= 500 and sentence:
                # Supprimer artefacts communs
                if not re.match(r'^(Figure|Table|Source|©)', sentence):
                    clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _calculate_sentence_positions(self, sentences: List[str], selected_indices: List[int]) -> List[float]:
        """Calcul scores position (biais vers début/fin)"""
        total = len(sentences)
        position_scores = []
        
        for idx in selected_indices:
            # Score position : élevé au début et fin
            if idx < total * 0.3:  # 30% début
                pos_score = 1.0
            elif idx > total * 0.7:  # 30% fin  
                pos_score = 0.8
            else:  # Milieu
                pos_score = 0.6
            
            position_scores.append(pos_score)
        
        return position_scores

class CamemBERTExtractor(ExtractiveModelBase):
    """Extracteur basé embeddings CamemBERT"""
    
    def __init__(self, model_name: str = 'dangvantuan/sentence-camembert-large'):
        super().__init__("camembert_extractive")
        self.model_name = model_name
        self.model = None
        self.fallback_models = [
            'dangvantuan/sentence-camembert-base',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/all-MiniLM-L6-v2'
        ]
    
    def load_model(self) -> bool:
        """Chargement CamemBERT avec fallbacks"""
        
        for model_name in [self.model_name] + self.fallback_models:
            try:
                logger.info(f" Chargement {model_name}...")
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name
                self.is_loaded = True
                logger.info(f" CamemBERT chargé: {model_name}")
                return True
                
            except Exception as e:
                logger.warning(f" Échec {model_name}: {e}")
                continue
        
        logger.error(" Tous les modèles CamemBERT ont échoué")
        return False
    
    def extract_sentences(self, text: str, num_sentences: int = 3, method: str = 'centroid') -> Dict:
        """Extraction par similarité sémantique"""
        
        if not self.is_loaded:
            raise RuntimeError("CamemBERT non chargé")
        
        start_time = time.time()
        
        try:
            sentences = self._split_sentences(text)
            
            if len(sentences) <= num_sentences:
                return {
                    'model': self.name,
                    'method': method,
                    'summary': '. '.join(sentences),
                    'selected_indices': list(range(len(sentences))),
                    'confidence': 0.9,
                    'processing_time': time.time() - start_time
                }
            
            # Génération embeddings
            embeddings = self.model.encode(sentences)
            
            if method == 'centroid':
                selected_indices = self._centroid_selection(embeddings, num_sentences)
            elif method == 'mmr':
                selected_indices = self._mmr_selection(embeddings, sentences, num_sentences)
            elif method == 'clustering':
                selected_indices = self._clustering_selection(embeddings, num_sentences)
            else:
                selected_indices = self._centroid_selection(embeddings, num_sentences)
            
            # Ordonner chronologiquement
            selected_indices = sorted(selected_indices)
            selected_sentences = [sentences[i] for i in selected_indices]
            
            # Métriques qualité
            confidence = self._calculate_extraction_confidence(embeddings, selected_indices)
            position_scores = self._calculate_sentence_positions(sentences, selected_indices)
            
            return {
                'model': self.name,
                'method': method,
                'summary': '. '.join(selected_sentences),
                'selected_indices': selected_indices,
                'selected_sentences': selected_sentences,
                'confidence': confidence,
                'position_scores': position_scores,
                'diversity_score': self._calculate_diversity(embeddings, selected_indices),
                'processing_time': time.time() - start_time,
                'model_variant': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Erreur extraction CamemBERT: {e}")
            return None
    
    def _centroid_selection(self, embeddings: np.ndarray, num_sentences: int) -> List[int]:
        """Sélection par proximité au centroïde"""
        
        # Centroïde global
        centroid = np.mean(embeddings, axis=0)
        
        # Similarités au centroïde
        similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        
        # Top-k indices
        top_indices = similarities.argsort()[-num_sentences:][::-1]
        
        return top_indices.tolist()
    
    def _mmr_selection(self, embeddings: np.ndarray, sentences: List[str], 
                      num_sentences: int, lambda_param: float = 0.7) -> List[int]:
        """Maximal Marginal Relevance pour diversité"""
        
        # Centroïde pour pertinence
        centroid = np.mean(embeddings, axis=0)
        relevance_scores = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
        
        selected = []
        remaining = list(range(len(sentences)))
        
        # Premier : plus pertinent
        first_idx = relevance_scores.argmax()
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Suivants : balance pertinence/diversité
        for _ in range(num_sentences - 1):
            mmr_scores = []
            
            for idx in remaining:
                # Score pertinence
                relevance = relevance_scores[idx]
                
                # Score diversité (1 - max similarité avec sélectionnés)
                if selected:
                    similarities_to_selected = cosine_similarity(
                        embeddings[idx].reshape(1, -1), 
                        embeddings[selected]
                    ).flatten()
                    max_similarity = similarities_to_selected.max()
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0
                
                # Score MMR
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                mmr_scores.append((mmr_score, idx))
            
            # Sélectionner le meilleur MMR
            best_score, best_idx = max(mmr_scores)
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def _clustering_selection(self, embeddings: np.ndarray, num_sentences: int) -> List[int]:
        """Sélection par clustering K-means"""
        
        from sklearn.cluster import KMeans
        
        # K-means clustering
        n_clusters = min(num_sentences, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        selected = []
        
        # Une phrase par cluster (la plus proche du centroïde)
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Plus proche du centroïde cluster
                cluster_centroid = kmeans.cluster_centers_[cluster_id]
                distances = cosine_similarity(cluster_embeddings, cluster_centroid.reshape(1, -1)).flatten()
                best_in_cluster = cluster_indices[distances.argmax()]
                selected.append(best_in_cluster)
        
        return selected[:num_sentences]
    
    def _calculate_extraction_confidence(self, embeddings: np.ndarray, selected: List[int]) -> float:
        """Confiance basée cohérence sémantique sélection"""
        
        if len(selected) < 2:
            return 0.8
        
        selected_embeddings = embeddings[selected]
        similarities = cosine_similarity(selected_embeddings)
        
        # Similarité moyenne entre phrases sélectionnées
        mean_similarity = np.triu(similarities, k=1).sum() / (len(selected) * (len(selected) - 1) / 2)
        
        # Confiance : ni trop similaire (redondance) ni trop différent (incohérence)
        optimal_similarity = 0.3  # Valeur empirique
        confidence = 1.0 - abs(mean_similarity - optimal_similarity) / optimal_similarity
        
        return max(0.3, min(1.0, confidence))
    
    def _calculate_diversity(self, embeddings: np.ndarray, selected: List[int]) -> float:
        """Score diversité sémantique"""
        
        if len(selected) < 2:
            return 1.0
        
        selected_embeddings = embeddings[selected]
        similarities = cosine_similarity(selected_embeddings)
        
        # Diversité = 1 - similarité moyenne
        mean_similarity = np.triu(similarities, k=1).sum() / (len(selected) * (len(selected) - 1) / 2)
        diversity = 1.0 - mean_similarity
        
        return max(0.0, min(1.0, diversity))

class TextRankExtractor(ExtractiveModelBase):
    """Extracteur basé TextRank (PageRank pour texte)"""
    
    def __init__(self):
        super().__init__("textrank")
    
    def load_model(self) -> bool:
        """TextRank ne nécessite pas de modèle externe"""
        self.is_loaded = True
        logger.info(" TextRank initialisé")
        return True
    
    def extract_sentences(self, text: str, num_sentences: int = 3, 
                         similarity_threshold: float = 0.1) -> Dict:
        """Extraction via algorithme TextRank"""
        
        start_time = time.time()
        
        try:
            sentences = self._split_sentences(text)
            
            if len(sentences) <= num_sentences:
                return {
                    'model': self.name,
                    'summary': '. '.join(sentences),
                    'selected_indices': list(range(len(sentences))),
                    'confidence': 0.8,
                    'processing_time': time.time() - start_time
                }
            
            # Construction matrice similarité
            similarity_matrix = self._build_similarity_matrix(sentences)
            
            # Algorithme PageRank
            scores = self._pagerank_scores(similarity_matrix, similarity_threshold)
            
            # Sélection top sentences
            top_indices = scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)  # Ordre chronologique
            
            selected_sentences = [sentences[i] for i in top_indices]
            
            return {
                'model': self.name,
                'method': 'textrank',
                'summary': '. '.join(selected_sentences),
                'selected_indices': top_indices.tolist(),
                'selected_sentences': selected_sentences,
                'confidence': self._calculate_textrank_confidence(scores, top_indices),
                'pagerank_scores': scores[top_indices].tolist(),
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Erreur TextRank: {e}")
            return None
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Construction matrice similarité basée TF-IDF"""
        
        # Vectorisation TF-IDF
        vectorizer = TfidfVectorizer(
            stop_words=['le', 'de', 'et', 'à', 'un', 'il', 'être', 'avoir', 'que', 'pour'],
            ngram_range=(1, 2),
            max_features=1000
        )
        
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def _pagerank_scores(self, similarity_matrix: np.ndarray, 
                        threshold: float = 0.1) -> np.ndarray:
        """Calcul scores PageRank"""
        
        # Seuillage similarité pour graphe sparse
        similarity_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
        
        # Construction graphe NetworkX
        graph = nx.from_numpy_array(similarity_matrix)
        
        # PageRank
        pagerank_scores = nx.pagerank(graph, max_iter=100, tol=1e-6)
        
        # Conversion array
        scores = np.array([pagerank_scores[i] for i in range(len(pagerank_scores))])
        
        return scores
    
    def _calculate_textrank_confidence(self, scores: np.ndarray, selected: List[int]) -> float:
        """Confiance basée distribution scores PageRank"""
        
        selected_scores = scores[selected]
        mean_score = selected_scores.mean()
        
        # Normalisation par score maximum
        max_score = scores.max()
        confidence = mean_score / max_score if max_score > 0 else 0.5
        
        return max(0.3, min(1.0, confidence))

class TFIDFExtractor(ExtractiveModelBase):
    """Extracteur basé TF-IDF avec MMR"""
    
    def __init__(self):
        super().__init__("tfidf_mmr")
        self.vectorizer = None
    
    def load_model(self) -> bool:
        """Initialisation TF-IDF vectorizer"""
        
        self.vectorizer = TfidfVectorizer(
            stop_words=['le', 'de', 'et', 'à', 'un', 'il', 'être', 'avoir', 'que', 'pour', 'dans', 'sur', 'avec'],
            ngram_range=(1, 3),
            max_features=2000,
            max_df=0.8,
            min_df=2
        )
        
        self.is_loaded = True
        logger.info(" TF-IDF extracteur initialisé")
        return True
    
    def extract_sentences(self, text: str, num_sentences: int = 3, 
                         lambda_mmr: float = 0.6) -> Dict:
        """Extraction TF-IDF + Maximal Marginal Relevance"""
        
        start_time = time.time()
        
        try:
            sentences = self._split_sentences(text)
            
            if len(sentences) <= num_sentences:
                return {
                    'model': self.name,
                    'summary': '. '.join(sentences),
                    'selected_indices': list(range(len(sentences))),
                    'confidence': 0.7,
                    'processing_time': time.time() - start_time
                }
            
            # Vectorisation TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Scores TF-IDF (somme par phrase)
            tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Sélection MMR
            selected_indices = self._mmr_selection_tfidf(
                tfidf_matrix, tfidf_scores, num_sentences, lambda_mmr
            )
            
            selected_indices = sorted(selected_indices)
            selected_sentences = [sentences[i] for i in selected_indices]
            
            return {
                'model': self.name,
                'method': 'tfidf_mmr',
                'summary': '. '.join(selected_sentences),
                'selected_indices': selected_indices,
                'selected_sentences': selected_sentences,
                'confidence': self._calculate_tfidf_confidence(tfidf_scores, selected_indices),
                'tfidf_scores': tfidf_scores[selected_indices].tolist(),
                'lambda_mmr': lambda_mmr,
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Erreur TF-IDF: {e}")
            return None
    
    def _mmr_selection_tfidf(self, tfidf_matrix, scores: np.ndarray, 
                            num_sentences: int, lambda_param: float) -> List[int]:
        """MMR avec représentations TF-IDF"""
        
        selected = []
        remaining = list(range(len(scores)))
        
        # Premier : score TF-IDF maximum
        first_idx = scores.argmax()
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Suivants : MMR
        for _ in range(num_sentences - 1):
            mmr_scores = []
            
            for idx in remaining:
                # Pertinence TF-IDF
                relevance = scores[idx]
                
                # Diversité (1 - max similarité avec sélectionnés)
                if selected:
                    similarities = cosine_similarity(
                        tfidf_matrix[idx], tfidf_matrix[selected]
                    ).flatten()
                    max_similarity = similarities.max()
                    diversity = 1 - max_similarity
                else:
                    diversity = 1.0
                
                # Score MMR
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                mmr_scores.append((mmr_score, idx))
            
            best_score, best_idx = max(mmr_scores)
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def _calculate_tfidf_confidence(self, scores: np.ndarray, selected: List[int]) -> float:
        """Confiance basée scores TF-IDF relatifs"""
        
        selected_scores = scores[selected]
        mean_selected = selected_scores.mean()
        mean_global = scores.mean()
        
        # Ratio vs moyenne globale
        confidence = min(mean_selected / mean_global, 2.0) / 2.0
        
        return max(0.3, min(1.0, confidence))

class ExtractiveFallback(ExtractiveModelBase):
    """Extracteur de fallback simple : premières phrases du texte"""
    
    def __init__(self):
        super().__init__("fallback_extractive")
    
    def load_model(self) -> bool:
        """Pas de modèle à charger pour le fallback"""
        self.is_loaded = True
        logger.info("✅ Extracteur fallback initialisé")
        return True
    
    def extract_sentences(self, text: str, num_sentences: int = 3) -> Dict:
        """Extraction simple des premières phrases significatives"""
        
        start_time = time.time()
        
        try:
            sentences = self._split_sentences(text)
            
            if not sentences:
                return {
                    'model': self.name,
                    'method': 'first_sentences',
                    'summary': '',
                    'selected_indices': [],
                    'selected_sentences': [],
                    'confidence': 0.1,
                    'processing_time': time.time() - start_time,
                    'fallback_reason': 'no_sentences_found'
                }
            
            # Sélection des premières phrases (jusqu'à num_sentences)
            selected_count = min(num_sentences, len(sentences))
            selected_indices = list(range(selected_count))
            selected_sentences = sentences[:selected_count]
            
            # Calcul confiance basé sur longueur et qualité des phrases
            confidence = self._calculate_fallback_confidence(selected_sentences)
            
            return {
                'model': self.name,
                'method': 'first_sentences',
                'summary': '. '.join(selected_sentences),
                'selected_indices': selected_indices,
                'selected_sentences': selected_sentences,
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'total_sentences_available': len(sentences),
                'fallback_reason': 'extractive_fallback'
            }
            
        except Exception as e:
            logger.error(f"Erreur fallback extractif: {e}")
            return {
                'model': self.name,
                'method': 'first_sentences',
                'summary': text[:500] + "..." if len(text) > 500 else text,
                'selected_indices': [],
                'selected_sentences': [],
                'confidence': 0.2,
                'processing_time': time.time() - start_time,
                'fallback_reason': 'emergency_truncation',
                'error': str(e)
            }
    
    def _calculate_fallback_confidence(self, sentences: List[str]) -> float:
        """Calcul confiance basé sur qualité des phrases sélectionnées"""
        
        if not sentences:
            return 0.1
        
        # Facteurs de qualité
        total_length = sum(len(s) for s in sentences)
        avg_length = total_length / len(sentences)
        
        # Confiance basée sur longueur moyenne des phrases
        if avg_length < 30:  # Phrases trop courtes
            confidence = 0.3
        elif avg_length > 200:  # Phrases trop longues
            confidence = 0.4
        else:  # Longueur acceptable
            confidence = 0.6
        
        # Bonus si plusieurs phrases disponibles
        if len(sentences) >= 2:
            confidence += 0.1
        
        # Pénalité si seulement 1 phrase
        if len(sentences) == 1:
            confidence -= 0.1
        
        return max(0.2, min(0.7, confidence))

class ExtractiveEnsemble:
    """Gestionnaire ensemble modèles extractifs"""
    
    def __init__(self):
        self.models = {}
        self.model_priorities = ['camembert', 'textrank', 'tfidf', 'fallback']
    
    def load_models(self, models_to_load: List[str] = None) -> Dict[str, bool]:
        """Chargement sélectif des modèles extractifs"""
        
        if models_to_load is None:
            models_to_load = self.model_priorities
        
        results = {}
        
        for model_name in models_to_load:
            try:
                if model_name == 'camembert':
                    model = CamemBERTExtractor()
                elif model_name == 'textrank':
                    model = TextRankExtractor()
                elif model_name == 'tfidf':
                    model = TFIDFExtractor()
                elif model_name == 'fallback':
                    model = ExtractiveFallback()
                else:
                    logger.warning(f"Modèle extractif inconnu: {model_name}")
                    results[model_name] = False
                    continue
                
                success = model.load_model()
                if success:
                    self.models[model_name] = model
                
                results[model_name] = success
                
            except Exception as e:
                logger.error(f"Erreur chargement {model_name}: {e}")
                results[model_name] = False
        
        logger.info(f"Modèles extractifs chargés: {list(self.models.keys())}")
        return results
    
    def extract_all_summaries(self, text: str, num_sentences: int = 3) -> List[Dict]:
        """Extraction avec tous les modèles disponibles"""
        
        summaries = []
        
        for model_name, model in self.models.items():
            try:
                # Paramètres spécialisés par modèle
                if model_name == 'camembert':
                    result = model.extract_sentences(text, num_sentences, method='mmr')
                elif model_name == 'textrank':
                    result = model.extract_sentences(text, num_sentences, similarity_threshold=0.15)
                elif model_name == 'tfidf':
                    result = model.extract_sentences(text, num_sentences, lambda_mmr=0.7)
                elif model_name == 'fallback':
                    result = model.extract_sentences(text, num_sentences)
                else:
                    result = model.extract_sentences(text, num_sentences)
                
                if result:
                    result['type'] = 'extractive'
                    summaries.append(result)
                    
            except Exception as e:
                logger.error(f"Erreur extraction {model_name}: {e}")
        
        return summaries
    
    def get_fallback_summary(self, text: str, num_sentences: int = 3) -> Dict:
        """Extraction de fallback garantie quand tous les autres modèles échouent"""
        
        # Essayer d'abord le fallback extractif si disponible
        if 'fallback' in self.models:
            try:
                result = self.models['fallback'].extract_sentences(text, num_sentences)
                if result and result.get('summary'):
                    result['type'] = 'extractive'
                    return result
            except Exception as e:
                logger.error(f"Erreur fallback extractif: {e}")
        
        # Fallback d'urgence si même le modèle fallback échoue
        try:
            fallback_model = ExtractiveFallback()
            fallback_model.load_model()
            result = fallback_model.extract_sentences(text, num_sentences)
            if result:
                result['type'] = 'extractive'
                result['emergency_fallback'] = True
                return result
        except Exception as e:
            logger.error(f"Erreur fallback d'urgence: {e}")
        
        # Dernier recours : troncature simple
        truncated_text = text[:500] + "..." if len(text) > 500 else text
        return {
            'model': 'emergency_truncation',
            'method': 'text_truncation',
            'summary': truncated_text,
            'type': 'extractive',
            'confidence': 0.1,
            'fallback_reason': 'all_extractive_models_failed',
            'emergency_fallback': True
        }
    
    def cleanup_all(self):
        """Nettoyage modèles"""
        self.models.clear()


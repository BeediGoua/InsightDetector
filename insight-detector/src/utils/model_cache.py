# src/utils/model_cache.py

from sentence_transformers import SentenceTransformer

# Cache global pour éviter de charger plusieurs fois les mêmes modèles
_ST_MODEL_CACHE = {}

def get_sentence_transformer(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Charge et met en cache un modèle SentenceTransformer.
    Usage : appeler get_sentence_transformer(model_name) partout dans le projet.
    """
    if model_name not in _ST_MODEL_CACHE:
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name]

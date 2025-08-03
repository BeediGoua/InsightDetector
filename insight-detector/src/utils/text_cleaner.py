# src/utils/text_cleaner.py
"""
Utilitaires de nettoyage de texte pour éliminer les métadonnées parasites
avant génération de résumés.
"""

import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """Nettoyeur de texte pour éliminer métadonnées et pollution"""
    
    def __init__(self):
        # Patterns pour détecter et supprimer les métadonnées
        self.metadata_patterns = [
            # Crédits photo/agence
            r'(?i)(?:photo|crédit|credits?)\s*:?\s*[A-Z\s/]+(?:AFP|REUTERS|SIPA|AP)',
            r'(?i)[A-Z\s]+/\s*(?:AFP|REUTERS|SIPA|AP|NURPHOTO)',
            r'(?i)(?:AFP|REUTERS|SIPA|AP)\s*/?\s*[A-Z\s]*',
            
            # Navigation web
            r'(?i)lire\s+plus\s+tard',
            r'(?i)google\s+actualit[eé]s?',
            r'(?i)partager\s+(?:cet\s+)?article\s*\??',
            r'(?i)masquer\s+ce\s+message',
            r'(?i)vous\s+souhaitez\s+partager\s+cet\s+article\s*\??',
            r'(?i)suivez[-\s]nous\s+sur',
            
            # Widgets sociaux
            r'(?i)abonnez[-\s]vous\s+à\s+notre\s+newsletter',
            r'(?i)pour\s+ne\s+(?:pas\s+)?manquer\s+aucune?\s+actualit[eé]',
            r'(?i)recevez\s+nos\s+alertes',
            r'(?i)t[eé]l[eé]chargez\s+notre\s+application',
            
            # Mentions légales/contact
            r'(?i)contact\s*@\s*\w+\.\w+',
            r'(?i)tous\s+droits\s+r[eé]serv[eé]s?',
            r'(?i)reproduction\s+interdite',
            
            # Timestamps et métadonnées d'article
            r'\d{1,2}\s+(?:janvier|f[eé]vrier|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|d[eé]cembre)\s+\d{4}',
            r'(?:publié|mis à jour)\s+le\s+\d{1,2}/\d{1,2}/\d{4}',
            
            # URLs et liens
            r'https?://[^\s]+',
            r'www\.[^\s]+',
            
            # Balises et caractères spéciaux
            r'<[^>]+>',  # Balises HTML
            r'&[a-zA-Z]+;',  # Entités HTML
            r'▁+',  # Artefacts SentencePiece
            
            # Légendes d'images
            r'(?i)(?:image|photo|illustration)\s*:?\s*[^\n]{0,100}(?:AFP|SIPA|REUTERS)',
        ]
        
        # Patterns pour détecter du contenu non-informatif
        self.noise_patterns = [
            r'(?i)(?:cliquez|cliquer)\s+(?:ici|sur|pour)',
            r'(?i)(?:voir|lire)\s+(?:aussi|également|plus)',
            r'(?i)(?:en\s+)?(?:savoir|apprendre)\s+plus',
            r'(?i)d[eé]couvrez?\s+(?:aussi|[eé]galement)',
        ]
    
    def clean_text(self, text: str, aggressive: bool = False) -> Optional[str]:
        """
        Nettoie un texte en supprimant les métadonnées parasites.
        
        Args:
            text: Texte à nettoyer
            aggressive: Si True, applique un nettoyage plus strict
            
        Returns:
            Texte nettoyé ou None si trop de contenu supprimé
        """
        if not text or not text.strip():
            return None
            
        original_length = len(text)
        cleaned = text
        
        # Supprimer métadonnées
        for pattern in self.metadata_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Supprimer bruit si mode agressif
        if aggressive:
            for pattern in self.noise_patterns:
                cleaned = re.sub(pattern, '', cleaned)
        
        # Normaliser espaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        # Vérifier si trop de contenu supprimé
        final_length = len(cleaned)
        if final_length < original_length * 0.3:  # Plus de 70% supprimé
            logger.warning(f"Nettoyage trop agressif: {original_length} → {final_length} caractères")
            return None
            
        if final_length < 100:  # Texte trop court après nettoyage
            logger.warning(f"Texte trop court après nettoyage: {final_length} caractères")
            return None
            
        return cleaned
    
    def validate_content_quality(self, text: str) -> dict:
        """
        Évalue la qualité du contenu avant résumé.
        
        Returns:
            Dict avec scores de qualité et recommandations
        """
        if not text:
            return {"valid": False, "reason": "Texte vide"}
            
        # Critères de qualité
        checks = {
            "length_ok": len(text) >= 100,
            "has_sentences": len(re.findall(r'[.!?]+', text)) >= 2,
            "not_mostly_metadata": self._count_metadata_ratio(text) < 0.5,
            "has_content_words": self._has_meaningful_content(text),
        }
        
        # Score global
        quality_score = sum(checks.values()) / len(checks)
        
        return {
            "valid": quality_score >= 0.75,
            "quality_score": quality_score,
            "checks": checks,
            "length": len(text),
            "sentences": len(re.findall(r'[.!?]+', text))
        }
    
    def _count_metadata_ratio(self, text: str) -> float:
        """Calcule la proportion de métadonnées dans le texte"""
        total_matches = 0
        for pattern in self.metadata_patterns:
            matches = re.findall(pattern, text)
            total_matches += sum(len(match) for match in matches)
        
        return total_matches / len(text) if text else 1.0
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Vérifie que le texte contient du contenu informatif"""
        content_indicators = [
            r'\b(?:selon|d\'après|affirme|déclare|annonce)\b',
            r'\b(?:recherche|étude|rapport|analyse)\b', 
            r'\b(?:entreprise|société|organisation)\b',
            r'\b(?:gouvernement|ministère|président)\b',
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in content_indicators)


# Fonction utilitaire pour intégration facile
def clean_text_for_summary(text: str, aggressive: bool = False) -> Optional[str]:
    """
    Fonction utilitaire pour nettoyer un texte avant résumé.
    
    Args:
        text: Texte à nettoyer
        aggressive: Nettoyage agressif
        
    Returns:
        Texte nettoyé ou None si non valide
    """
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(text, aggressive=aggressive)
    
    if cleaned:
        quality = cleaner.validate_content_quality(cleaned)
        if quality["valid"]:
            return cleaned
        else:
            logger.warning(f"Texte rejeté: qualité {quality['quality_score']:.2f}")
            return None
    
    return None
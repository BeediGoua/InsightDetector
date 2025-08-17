# src/detection/level3_improvement/level3_utils.py
# -------------------------------------------------
# Utilitaires Level 3 : IO JSONL, hashing, détection langue,
# chunking, post-traitement, choix de mode, génération (EDIT/RESUM),
# évaluation L2 simplifiée (placeholder) et critères d'acceptation.
#
# Pour activer des modèles HuggingFace :
#   export L3_USE_HF=1
# (sinon generate_edit / generate_resummarize utilisent un fallback naïf.)

from __future__ import annotations
import os, re, json, hashlib, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ============================================================
# ---------------------- IO HELPERS --------------------------
# ============================================================

def sha1_text(s: str) -> str:
    """Hash SHA1 robuste pour cache/déduplication."""
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Lecture JSONL. Retourne [] si le fichier n'existe pas."""
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Écriture JSONL (création des dossiers si besoin)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============================================================
# -------------------- LANG & TEXT OPS -----------------------
# ============================================================

def _as_text(x: Any) -> str:
    """Convertit n'importe quelle valeur en str sûre (gère None/NaN/float)."""
    if isinstance(x, str):
        return x
    try:
        import pandas as pd  # facultatif
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return "" if x is None else str(x)

_FR_STOPS = {"le","la","les","des","du","de","un","une","et","pour","avec","dans","sur","par","au","aux","est","sont","été"}
_EN_STOPS = {"the","of","and","to","in","for","on","with","by","is","are","was","were","as","at","from"}

def detect_lang(text: str) -> str:
    """Heuristique FR/EN simple (suffisant pour router les modèles)."""
    t = (text or "").lower()
    fr_score = sum(w in t for w in _FR_STOPS)
    en_score = sum(w in t for w in _EN_STOPS)
    return "fr" if fr_score >= en_score else "en"

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
_WS = re.compile(r"\s+")

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

def clean_text(text: str) -> str:
    """Nettoyage minimal : espaces multiples, espaces avant ponctuation, trim."""
    t = (text or "").strip()
    t = re.sub(r"\s+([,;:\.\!\?])", r"\1", t)
    t = _WS.sub(" ", t)
    return t.strip()

def enforce_length(text: str, min_w: int = 70, max_w: int = 120) -> str:
    """Force la longueur entre min_w et max_w mots (tronque ou allonge légèrement)."""
    words = (text or "").split()
    if len(words) > max_w:
        return " ".join(words[:max_w])
    if len(words) < min_w and words:
        pad = words[-12:] if len(words) >= 12 else words
        aug = words + pad
        return " ".join(aug[:max_w])
    return text or ""

def postprocess_summary(text: str, min_w: int = 70, max_w: int = 120) -> str:
    """Nettoyage + normalisation de longueur."""
    return enforce_length(clean_text(text), min_w=min_w, max_w=max_w)

def chunk_text_by_words(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Découpage par mots (≈ 800–1200 tokens) avec overlap léger."""
    words = (text or "").split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

# ============================================================
# --------------- MODE (EDIT vs RE-SUMMARIZE) ----------------
# ============================================================

def choose_mode(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, str, Dict[str, bool]]:
    """
    Décide 'edit' vs 're_summarize' selon nos règles assouplies :
      - CW & CRITICAL -> re_summarize si has_text (seuil abaissé), sinon edit
      - Adaptive & CRITICAL -> edit si issues<=8 & factuality>=0.75, sinon re_summarize si texte ok, sinon edit
      - Sinon -> edit
    Retourne (mode, mode_reason, flags).
    """
    has_text = bool(row.get("has_text"))
    enough_length = bool(row.get("enough_length"))
    strat = row.get("strategy")
    tier  = row.get("tier")
    issues = float(row.get("issues_count") or 0.0)
    factuality = float(row.get("factuality_score") or 0.0)

    # Seuil abaissé pour enough_length - utiliser min_text_chars_for_resummarize
    min_chars = cfg.get("min_text_chars_for_resummarize", 500)
    text_length = len(str(row.get("text", "")))
    enough_length_relaxed = has_text and text_length >= min_chars

    flags = {
        "has_text": has_text,
        "enough_length": enough_length_relaxed,  # Utiliser le seuil relaxé
        "lang": row.get("lang") or "fr"
    }

    mode = "edit"
    reason = "default_edit"

    if tier == "CRITICAL" and strat == "confidence_weighted" and cfg["mode"]["prefer_resum_for_cw_critical"]:
        if has_text and enough_length_relaxed:  # Seuil abaissé
            mode, reason = "re_summarize", "cw_critical_with_text"
        else:
            mode, reason = "edit", "cw_critical_no_text"

    elif tier == "CRITICAL" and strat == "adaptive":
        if issues <= cfg["edit_rule_adaptive"]["issues_max"] and factuality >= cfg["edit_rule_adaptive"]["factuality_min"]:
            mode, reason = "edit", "adaptive_edit_rule"
        else:
            if has_text and enough_length_relaxed:  # Seuil abaissé
                mode, reason = "re_summarize", "adaptive_critical_with_text"
            else:
                mode, reason = "edit", "adaptive_critical_no_text"

    return mode, reason, flags

# ============================================================
# ------------------- GENERATION (PLACEHOLDER) ---------------
# ============================================================

def _placeholder_edit(summary_before: str, seed: int = 42) -> str:
    """Édition naïve : sélection de 3–4 phrases distinctes puis nettoyage."""
    random.seed(seed)
    sents = split_sentences(summary_before)
    if not sents:
        return summary_before or ""
    chosen: List[str] = []
    for s in sents:
        if not any(s.lower() in c.lower() or c.lower() in s.lower() for c in chosen):
            chosen.append(s)
        if len(chosen) >= 4:
            break
    out = " ".join(chosen)
    return postprocess_summary(out)

def _placeholder_resum(source_text: str, seed: int = 42) -> str:
    """Résumé extractif naïf : premières phrases jusqu'à ~100 mots."""
    random.seed(seed)
    sents = split_sentences(source_text)
    out, w = [], 0
    for s in sents:
        out.append(s)
        w += len(s.split())
        if w >= 100:
            break
    return postprocess_summary(" ".join(out))

# ============================================================
# --------- GENERATION (LLM HuggingFace optionnelle) ---------
# ============================================================

_USE_HF = os.getenv("L3_USE_HF", "0").strip().lower() in {"1","true","yes","y"}
_EDIT_MODEL_ID = os.getenv("L3_EDIT_MODEL", "google/flan-t5-base")
_RESUM_FR_ID   = os.getenv("L3_RESUM_FR_MODEL", "csebuetnlp/mT5_multilingual_XLSum")
_RESUM_EN_ID   = os.getenv("L3_RESUM_EN_MODEL", "facebook/bart-large-cnn")

_hf_ok = False
_edit_pipe = None
_resum_fr_pipe = None
_resum_en_pipe = None

if _USE_HF:
    try:
        from transformers import pipeline
        _hf_ok = True
    except Exception:
        _hf_ok = False

def _safe_gen_params(max_new_tokens: int = 220) -> Dict[str, Any]:
    """Paramètres anti-hallucinations (≈ temperature 0)."""
    return dict(
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        max_new_tokens=max_new_tokens
    )

def _apply_stops(text: str, stops: List[str]) -> str:
    """Coupe le texte au premier stop token rencontré."""
    text = text or ""
    for s in stops:
        if s and s in text:
            return text.split(s)[0].strip()
    return text.strip()

def _get_edit_pipe():
    global _edit_pipe
    if _edit_pipe is None:
        _edit_pipe = pipeline(task="text2text-generation", model=_EDIT_MODEL_ID)
    return _edit_pipe

def _get_resum_pipe(lang: str):
    global _resum_fr_pipe, _resum_en_pipe
    if lang == "fr":
        if _resum_fr_pipe is None:
            _resum_fr_pipe = pipeline(task="summarization", model=_RESUM_FR_ID)
        return _resum_fr_pipe
    else:
        if _resum_en_pipe is None:
            _resum_en_pipe = pipeline(task="summarization", model=_RESUM_EN_ID)
        return _resum_en_pipe

# ============================================================
# -------------------- API de génération ---------------------
# ============================================================

def generate_edit(summary_before: str, seed: int = 42, lang: str = "fr") -> str:
    """
    Réécriture contrainte (EDIT).
    - Sans LLM : nettoyage/sélection de phrases.
    - Avec LLM : FLAN-T5 par prompt directif (factualité stricte).
    """
    summary_before = _as_text(summary_before)
    if not summary_before:
        return ""
    if _USE_HF and _hf_ok:
        try:
            pipe = _get_edit_pipe()
            if lang == "fr":
                prompt = (
                    "Réécris ce résumé en CONSERVANT STRICTEMENT les faits, "
                    "sans ajout externe. Supprime répétitions et ambiguïtés. "
                    "70–120 mots. Style factuel, clair, phrases courtes.\n\n"
                    f"Résumé:\n{summary_before}\n\nRéécriture:"
                )
            else:
                prompt = (
                    "Rewrite this summary while STRICTLY PRESERVING the facts, "
                    "no external knowledge. Remove repetitions/ambiguities. "
                    "70–120 words. Factual, clear, short sentences.\n\n"
                    f"Summary:\n{summary_before}\n\nRewrite:"
                )
            out = pipe(prompt, **_safe_gen_params(max_new_tokens=220))[0]["generated_text"]
            out = _apply_stops(out, ["\n\n", "###"])
            return postprocess_summary(out, 70, 120)
        except Exception:
            pass  # fallback si HF échoue
    return _placeholder_edit(summary_before, seed=seed)

def generate_resummarize(source_text: str, seed: int = 42, lang: str = "fr") -> str:
    """
    Re-summarization (depuis texte source).
    - Sans LLM : extractif naïf.
    - Avec LLM : mT5(XLSum) en FR / BART-CNN en EN.
    """
    source_text = _as_text(source_text)
    if not source_text:
        return ""
    if _USE_HF and _hf_ok:
        try:
            pipe = _get_resum_pipe(lang)
            out = pipe(source_text, truncation=True, **_safe_gen_params(max_new_tokens=220))[0]["summary_text"]
            out = _apply_stops(out, ["\n\n", "###"])
            return postprocess_summary(out, 70, 120)
        except Exception:
            pass  # fallback si HF échoue
    return _placeholder_resum(source_text, seed=seed)

# ============================================================
# --------------------- ÉVALUATION L2 MOCK -------------------
# ============================================================

def _issues_heuristic(text: str) -> int:
    t = text or ""
    issues = 0
    issues += t.count("??") + t.count("?!")
    issues += t.count("  ")
    issues += (1 if len(t.strip()) == 0 else 0)
    issues += (1 if re.search(r"\([^\)]*$", t) else 0)  # parenthèse non fermée
    return max(0, issues)

def _coherence_heuristic(text: str) -> float:
    sents = split_sentences(text)
    if not sents:
        return 0.0
    avg_len = sum(len(s.split()) for s in sents) / max(1, len(sents))
    score = 1.0
    score -= max(0, abs(avg_len - 25) / 40)               # pénalité longueur
    score -= 0.1 * sum(1 for s in sents if not s[:1].istitle())  # style
    return max(0.0, min(1.0, score))

def _factuality_heuristic(summary: str, source_text: str = "") -> float:
    summary = _as_text(summary)
    source_text = _as_text(source_text)

    if not summary:
        return 0.0
    if not source_text:
        # Sans source, rester conservateur (EDIT sans article)
        return 0.85

    sum_words = set(w.lower() for w in re.findall(r"\w+", summary))
    src_words = set(w.lower() for w in re.findall(r"\w+", source_text))
    if not sum_words:
        return 0.0
    overlap = len(sum_words & src_words) / max(1, len(sum_words))
    # comprime la plage vers ~0.70–0.98
    return max(0.70, min(0.98, 0.70 + 0.50 * overlap))

def l2_like_evaluate(summary_after: str, source_text: str = "") -> Dict[str, Any]:
    """Éval simplifiée (tier/factuality/coherence/issues) pour la boucle de test."""
    summary_after = _as_text(summary_after)
    source_text = _as_text(source_text)

    issues = _issues_heuristic(summary_after)
    coherence = _coherence_heuristic(summary_after)
    factuality = _factuality_heuristic(summary_after, source_text)

    if factuality >= 0.95 and coherence >= 0.90 and issues <= 1:
        tier = "EXCELLENT"
    elif factuality >= 0.90 and coherence >= 0.80 and issues <= 2:
        tier = "GOOD"
    elif factuality >= 0.80 and coherence >= 0.60 and issues <= 5:
        tier = "MODERATE"
    else:
        tier = "CRITICAL"

    return {
        "tier": tier,
        "factuality_score": round(factuality, 3),
        "coherence_score": round(coherence, 3),
        "issues_count": int(issues)
    }

# ============================================================
# --------------------- ACCEPTANCE RULES ---------------------
# ============================================================

def accept_after(before: Dict[str, Any], after: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Critères d'acceptation assouplies :
      - tier_after ∈ {GOOD, EXCELLENT, MODERATE} OU CRITICAL gardé avec amélioration
      - OU amélioration significative (issues, factuality, coherence)
    """
    acc = cfg["acceptance"]
    ok_tier = after["tier"] in acc["accepted_tiers"]
    
    # Acceptation IMPROVED_CRITICAL si amélioration notable
    if not ok_tier and after["tier"] == "CRITICAL" and acc.get("allow_critical_with_improvement", True):
        critical_guard = acc["critical_guard"]
        
        # Critères CRITICAL avec amélioration
        issues_improved = after["issues_count"] <= critical_guard["issues_max"]
        factuality_ok = after["factuality_score"] >= critical_guard["factuality_min"]
        coherence_ok = after["coherence_score"] >= critical_guard["coherence_min"]
        
        # Amélioration relative vs before
        issues_reduced = after["issues_count"] < before.get("issues_count", 999)
        factuality_improved = after["factuality_score"] >= before.get("factuality_score", 0) + critical_guard["improvement_required"]
        coherence_improved = after["coherence_score"] >= before.get("coherence_score", 0) + critical_guard["improvement_required"]
        
        # Accepter si critères absolus OK OU amélioration relative
        if (issues_improved and factuality_ok and coherence_ok) or issues_reduced or factuality_improved or coherence_improved:
            ok_tier = True
    
    # Garde-fou MODERATE (assouplies)
    if not ok_tier and acc.get("allow_moderate_guarded", True):
        if (after["tier"] == "MODERATE"
            and after["issues_count"] <= acc["moderate_guard"]["issues_max"]
            and after["factuality_score"] >= acc["moderate_guard"]["factuality_min"]
            and after["coherence_score"]  >= acc["moderate_guard"]["coherence_min"]):
            ok_tier = True
                
    if not ok_tier:
        return False, f"tier_after={after['tier']} not accepted"

    # Amélioration monotone optionnelle (désactivée par défaut pour CRITICAL)
    if acc.get("require_monotonic_improvement", False):
        if float(after["factuality_score"]) < float(before.get("factuality_score", 0) or 0):
            return False, "factuality not improved"
        if float(after["coherence_score"])  < float(before.get("coherence_score", 0) or 0):
            return False, "coherence not improved"
    
    # Garde-fou pour stagnation si réduction d'issues
    elif acc.get("allow_stagnation_if_issues_reduced", True):
        if after["issues_count"] < before.get("issues_count", 999):
            return True, "accepted - issues reduced"

    return True, "accepted"

# ============================================================
# ------------------------- EXPORTS --------------------------
# ============================================================

__all__ = [
    "sha1_text",
    "read_jsonl", "write_jsonl",
    "_as_text",
    "detect_lang", "chunk_text_by_words",
    "postprocess_summary",
    "choose_mode",
    "generate_edit", "generate_resummarize",
    "l2_like_evaluate",
    "accept_after",
]

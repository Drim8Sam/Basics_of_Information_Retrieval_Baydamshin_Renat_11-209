from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set

STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could",
    "did", "do", "does", "doing", "down", "during",
    "each",
    "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself",
    "just",
    "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own",
    "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
    # частые HTML-энтити
    "nbsp", "amp", "lt", "gt", "quot", "apos",
}

TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE | re.ASCII)

def load_doc_urls(index_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not index_path.exists():
        return mapping

    for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", maxsplit=1)
        if len(parts) == 2:
            try:
                mapping[int(parts[0])] = parts[1]
            except ValueError:
                continue

    return mapping


def load_document_vectors(
    tfidf_dir: Path,
    mode: str = "lemmas",
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    suffix = f"_{mode}.txt"
    doc_vectors: Dict[int, Dict[str, float]] = {}
    global_idf: Dict[str, float] = {}

    for fpath in sorted(tfidf_dir.glob(f"*{suffix}")):
        try:
            doc_id = int(fpath.stem.replace(f"_{mode}", ""))
        except ValueError:
            continue

        vec: Dict[str, float] = {}
        for line in fpath.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            term = parts[0]
            try:
                idf = float(parts[1])
                tfidf = float(parts[2])
            except ValueError:
                continue
            vec[term] = tfidf
            global_idf[term] = idf

        doc_vectors[doc_id] = vec

    return doc_vectors, global_idf

def _is_noise(tok: str) -> bool:
    if not tok.isascii():
        return True
    if len(tok) >= 5 and len(set(tok)) == 1:
        return True
    if len(tok) > 30:
        return True
    return False


def tokenize_query(text: str, min_length: int = 2) -> List[str]:
    tokens: List[str] = []
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0).lower()
        if len(tok) < min_length:
            continue
        if tok in STOP_WORDS:
            continue
        if not tok.isalpha():
            continue
        if _is_noise(tok):
            continue
        tokens.append(tok)
    return tokens


def get_stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


def build_query_vector(
    query_tokens: List[str],
    global_idf: Dict[str, float],
    stemmer=None,
) -> Dict[str, float]:
    if not query_tokens:
        return {}

    processed: List[str] = []
    for tok in query_tokens:
        if stemmer is not None:
            processed.append(stemmer.stem(tok))
        else:
            processed.append(tok)

    total = len(processed)
    counts = Counter(processed)

    vector: Dict[str, float] = {}
    for term, cnt in counts.items():
        idf = global_idf.get(term)
        if idf is None:
            continue
        tf = cnt / total
        vector[term] = tf * idf

    return vector


def _norm(vec: Dict[str, float]) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot = 0.0
    for key, val_a in vec_a.items():
        val_b = vec_b.get(key)
        if val_b is not None:
            dot += val_a * val_b

    if dot == 0.0:
        return 0.0

    norm_a = _norm(vec_a)
    norm_b = _norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)

def search(
    query: str,
    doc_vectors: Dict[int, Dict[str, float]],
    global_idf: Dict[str, float],
    stemmer=None,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    tokens = tokenize_query(query)
    if not tokens:
        return []

    query_vec = build_query_vector(tokens, global_idf, stemmer=stemmer)
    if not query_vec:
        return []

    scores: List[Tuple[int, float]] = []
    for doc_id, doc_vec in doc_vectors.items():
        sim = cosine_similarity(query_vec, doc_vec)
        if sim > 0.0:
            scores.append((doc_id, sim))

    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores[:top_k]

